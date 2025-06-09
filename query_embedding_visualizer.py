import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import umap
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import json
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Query Embedding Visualizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Claude-inspired styling
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        font-size: 18px;
        line-height: 1.6;
        color: #2d2d2d;
        background-color: #ffffff;
    }
    p, div, span, li, td, th { font-size: max(18px, 1rem) !important; }
    h1 { font-size: 32px !important; font-weight: 600 !important; color: #1a1a1a !important; margin-bottom: 0.5rem !important; }
    h2 { font-size: 26px !important; font-weight: 600 !important; color: #1a1a1a !important; }
    h3 { font-size: 22px !important; font-weight: 600 !important; color: #1a1a1a !important; }
    .main { padding: 2rem; background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #fafafa; border-right: 1px solid #e5e5e5; }
    .stButton > button {
        background-color: #dc6b2f; color: white; font-weight: 500; border-radius: 6px; border: none;
        padding: 0.75rem 1.5rem; font-size: 18px !important; transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #c55a24; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(220, 107, 47, 0.2);
    }
    .secondary-button > button { background-color: #ffffff; color: #2d2d2d; border: 1px solid #e5e5e5; }
    .secondary-button > button:hover { background-color: #fafafa; border-color: #dc6b2f; color: #dc6b2f; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; background-color: #f4f4f4; padding: 4px; border-radius: 8px; }
    .stTabs [data-baseweb="tab"] { font-size: 18px !important; font-weight: 500; color: #666666; background-color: transparent; border-radius: 6px; padding: 8px 20px; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { background-color: white; color: #1a1a1a; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea { font-size: 18px !important; border: 1px solid #e5e5e5; border-radius: 6px; padding: 0.75rem; }
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus { border-color: #dc6b2f; box-shadow: 0 0 0 1px #dc6b2f; }
    .stSelectbox > div > div { font-size: 18px !important; }
    [data-testid="metric-container"] { background-color: #fafafa; border: 1px solid #e5e5e5; padding: 1.25rem; border-radius: 6px; }
    .info-box { background-color: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; padding: 1.5rem; position: sticky; top: 2rem; }
    .chunk-container { background-color: #fafafa; border: 1px solid #e5e5e5; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
    .stHelp { font-size: 16px !important; color: #666666; }
    .streamlit-expanderHeader { background-color: #fafafa; border: 1px solid #e5e5e5; border-radius: 6px; font-size: 18px !important; font-weight: 500; color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'content_chunks' not in st.session_state:
    st.session_state.content_chunks = []
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}
if 'selected_vector' not in st.session_state:
    st.session_state.selected_vector = None

# Initialize Gemini with API key from environment
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found in environment variables. Please check your .env file.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Query type color mapping (matching Fan-Out Simulator)
QUERY_TYPE_COLORS = {
    "reformulation": "#0066cc",
    "related": "#6600cc",
    "implicit": "#008800",
    "comparative": "#cc6600",
    "entity_expansion": "#cc0066",
    "personalized": "#008866",
    "head_query": "#dc6b2f",  # Orange for head query
    "content": ["#4a5568", "#2d3748", "#1a202c", "#171923"]  # Shades of gray for content
}

def get_query_color(index):
    """Generate unique colors for queries"""
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
        '#FF9FF3', '#54A0FF', '#48DBFB', '#1DD1A1', '#FFC048',
        '#FF6B9D', '#C44569', '#F8B500', '#00B894', '#6C5CE7',
        '#A29BFE', '#FD79A8', '#FDCB6E', '#6D5D6E', '#4F4557'
    ]
    return colors[index % len(colors)]

def get_embeddings(texts):
    """Generate embeddings using Gemini API"""
    embeddings = []
    for text in texts:
        # Check cache first
        if text in st.session_state.embeddings_cache:
            embeddings.append(st.session_state.embeddings_cache[text])
        else:
            try:
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document",
                    title="Query Embedding"
                )
                embedding = np.array(result['embedding'])
                st.session_state.embeddings_cache[text] = embedding
                embeddings.append(embedding)
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                st.error(f"Error generating embedding: {e}")
                embeddings.append(np.zeros(768))
    return np.array(embeddings)

def add_content_chunk():
    """Add a new content chunk"""
    if len(st.session_state.content_chunks) < 20:
        st.session_state.content_chunks.append({
            'id': len(st.session_state.content_chunks) + 1,
            'label': '',
            'content': '',
            'target_query': ''
        })
    else:
        st.warning("Maximum of 20 content chunks reached")

def remove_content_chunk(chunk_id):
    """Remove a content chunk"""
    st.session_state.content_chunks = [
        chunk for chunk in st.session_state.content_chunks 
        if chunk['id'] != chunk_id
    ]
    # Renumber chunks
    for i, chunk in enumerate(st.session_state.content_chunks):
        chunk['id'] = i + 1

def generate_cluster_names(embeddings, labels, n_clusters):
    """Generate meaningful names for clusters based on their content"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    cluster_names = {}
    for i in range(n_clusters):
        cluster_texts = [labels[j] for j in range(len(labels)) if cluster_labels[j] == i]
        if cluster_texts:
            cluster_names[i] = f"Cluster {i+1}: {cluster_texts[0][:30]}..."
        else:
            cluster_names[i] = f"Cluster {i+1}"
    return cluster_labels, cluster_names

# Header
st.title("Query Embedding Visualizer")
st.markdown("Visualize how queries and content relate in semantic space")

# Main layout
col1, col2 = st.columns([3, 1])

with col1:
    tab1, tab2, tab3 = st.tabs(["Query Input", "Content Chunks", "Import Data"])
    with tab1:
        head_query = st.text_input(
            "Head Query", 
            value="how to find safe and kind dentist for my child",
            help="The main query to compare everything against"
        )
        sub_queries_input = st.text_area(
            "Simulated Sub-queries",
            height=200,
            help="Enter one query per line (or import from CSV)",
            placeholder="Enter your simulated queries here, one per line..."
        )
        query_types_input = st.text_area(
            "Query Types (Optional)",
            height=100,
            help="Enter query types matching the order of sub-queries (reformulation, related, implicit, comparative, entity_expansion, personalized)"
        )
    with tab2:
        # Collapsible by default!
        with st.expander("Content Chunks", expanded=False):
            st.markdown("Add content chunks to see how they relate to your queries")
            for i, chunk in enumerate(st.session_state.content_chunks):
                with st.container():
                    st.markdown(f"<div class='chunk-container'>", unsafe_allow_html=True)
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        st.markdown(f"**Chunk {chunk['id']}**")
                        chunk['label'] = st.text_input(
                            "Label",
                            value=chunk['label'],
                            key=f"label_{chunk['id']}",
                            placeholder="e.g., About pediatric dental care"
                        )
                        if sub_queries_input:
                            queries = [q.strip() for q in sub_queries_input.strip().split('\n') if q.strip()]
                            chunk['target_query'] = st.selectbox(
                                "Target Query",
                                options=[''] + queries,
                                key=f"target_{chunk['id']}",
                                help="Which query is this content targeting?"
                            )
                        chunk['content'] = st.text_area(
                            "Content",
                            value=chunk['content'],
                            key=f"content_{chunk['id']}",
                            height=100,
                            placeholder="Enter your content chunk here..."
                        )
                    with col_b:
                        st.markdown("<br>", unsafe_allow_html=True)
                        if st.button("Remove", key=f"remove_{chunk['id']}"):
                            remove_content_chunk(chunk['id'])
                            st.rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
            if st.button("+ Add Content Chunk", type="secondary"):
                add_content_chunk()
                st.rerun()
    with tab3:
        st.markdown("### Import Template")
        st.markdown("Download this template, fill it with your queries, and upload it back.")
        
        # Create template
        template_df = pd.DataFrame({
            'query': [
                'how to find safe dentist for children',
                'pediatric dentist near me',
                'child friendly dental care',
                'what to look for in kids dentist',
                'dental anxiety in children'
            ],
            'type': [
                'reformulation',
                'related',
                'implicit',
                'comparative',
                'entity_expansion'
            ]
        })
        
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=template_csv,
            file_name="query_template.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Import CSV from Query Fan-Out Simulator",
            type="csv",
            help="Upload the exported CSV from the Query Fan-Out Simulator or use the template above"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'query' in df.columns:
                    queries = df['query'].tolist()
                    types = df['type'].tolist() if 'type' in df.columns else []
                    sub_queries_input = '\n'.join(queries)
                    query_types_input = '\n'.join(types) if types else ''
                    st.success(f"Imported {len(queries)} queries")
                else:
                    st.error("CSV must contain a 'query' column")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    st.markdown("---")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
    with col_ctrl1:
        projection_mode = st.radio("View Mode", ["3D", "2D"], horizontal=True)
    with col_ctrl2:
        st.empty()  # Removed similarity threshold checkbox
    with col_ctrl3:
        search_query = st.text_input("Search vectors", placeholder="Type to search...")
    if st.button("Generate Visualization", type="primary", use_container_width=True):
        if head_query and sub_queries_input:
            with st.spinner("Generating embeddings..."):
                all_texts = [head_query]
                all_labels = ["Head Query"]
                all_types = ["head_query"]
                sub_queries = [q.strip() for q in sub_queries_input.strip().split('\n') if q.strip()]
                query_types = [t.strip() for t in query_types_input.strip().split('\n') if t.strip()] if query_types_input else []
                if len(query_types) < len(sub_queries):
                    query_types.extend(['related'] * (len(sub_queries) - len(query_types)))
                all_texts.extend(sub_queries)
                all_labels.extend([f"Query: {q[:50]}..." if len(q) > 50 else f"Query: {q}" for q in sub_queries])
                all_types.extend(query_types[:len(sub_queries)])
                valid_chunks = [c for c in st.session_state.content_chunks if c['content'].strip()]
                for chunk in valid_chunks:
                    all_texts.append(chunk['content'])
                    label = chunk['label'] or f"Chunk {chunk['id']}"
                    all_labels.append(f"Content: {label}")
                    all_types.append('content')
                embeddings = get_embeddings(all_texts)
                n_components = 3 if projection_mode == "3D" else 2
                reducer = umap.UMAP(n_components=n_components, random_state=42, min_dist=0.3)
                embeddings_reduced = reducer.fit_transform(embeddings)
                n_clusters = min(5, len(all_texts) // 3) if len(all_texts) > 3 else 1
                cluster_labels, cluster_names = generate_cluster_names(embeddings, all_labels, n_clusters)
                head_embedding = embeddings[0].reshape(1, -1)
                similarities = cosine_similarity(head_embedding, embeddings)[0]
                # Add embedding dimensions to dataframe
                embedding_cols = [f'dim_{i}' for i in range(embeddings.shape[1])]
                viz_df = pd.DataFrame({
                    'x': embeddings_reduced[:, 0],
                    'y': embeddings_reduced[:, 1],
                    'z': embeddings_reduced[:, 2] if projection_mode == "3D" else 0,
                    'text': all_texts,
                    'label': all_labels,
                    'type': all_types,
                    'cluster': cluster_labels,
                    'cluster_name': [cluster_names[c] for c in cluster_labels],
                    'similarity_to_head': similarities
                })
                for i, col in enumerate(embedding_cols):
                    viz_df[col] = embeddings[:, i]
                st.session_state.viz_df = viz_df
                st.session_state.embeddings = embeddings
                st.session_state.cluster_names = cluster_names
        else:
            st.error("Please enter a head query and at least one sub-query")

# Visualization display
if 'viz_df' in st.session_state:
    df = st.session_state.viz_df
    
    # Create two columns for visualization and cluster navigation
    viz_col, nav_col = st.columns([3, 1])
    
    with nav_col:
        st.markdown("### Cluster Navigation")
        selected_cluster = st.selectbox(
            "Select Cluster",
            options=['All'] + list(st.session_state.cluster_names.values()),
            key='cluster_nav'
        )
        
        # Get cluster index from name
        selected_cluster_idx = None
        if selected_cluster != 'All':
            for idx, name in st.session_state.cluster_names.items():
                if name == selected_cluster:
                    selected_cluster_idx = idx
                    break
    
    with viz_col:
        fig = go.Figure()
        
        for vec_type in df['type'].unique():
            type_df = df[df['type'] == vec_type]
            if vec_type == 'head_query':
                color = QUERY_TYPE_COLORS['head_query']
                symbol = 'diamond'
                size = 15
            elif vec_type == 'content':
                colors = []
                for i, row in type_df.iterrows():
                    chunk_idx = i - len(df[df['type'] != 'content'])
                    color_idx = chunk_idx % len(QUERY_TYPE_COLORS['content'])
                    colors.append(QUERY_TYPE_COLORS['content'][color_idx])
                symbol = 'square'
                size = 12
            else:
                # Generate unique colors for each query
                colors = []
                base_idx = len(df[df['type'] == 'head_query'])
                for i, row in type_df.iterrows():
                    query_idx = i - base_idx
                    colors.append(get_query_color(query_idx))
                symbol = 'circle'
                size = 8  # Smaller size for queries
            
            hover_text = []
            for _, row in type_df.iterrows():
                text = f"<b>{row['label']}</b><br>"
                text += f"Type: {row['type']}<br>"
                text += f"Cluster: {row['cluster_name']}<br>"
                text += f"Similarity to Head: {row['similarity_to_head']:.3f}<br>"
                text += f"<br>Text: {row['text'][:100]}..."
                hover_text.append(text)
            
            if projection_mode == "3D":
                trace = go.Scatter3d(
                    x=type_df['x'],
                    y=type_df['y'],
                    z=type_df['z'],
                    mode='markers',
                    name=vec_type.replace('_', ' ').title(),
                    marker=dict(
                        size=size,
                        color=colors if vec_type in ['content', 'reformulation', 'related', 'implicit', 'comparative', 'entity_expansion', 'personalized'] else color,
                        symbol=symbol,
                        line=dict(width=2, color='white') if vec_type == 'head_query' else dict(width=1)
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    hovertemplate='%{text}<extra></extra>'
                )
            else:
                trace = go.Scatter(
                    x=type_df['x'],
                    y=type_df['y'],
                    mode='markers',
                    name=vec_type.replace('_', ' ').title(),
                    marker=dict(
                        size=size,
                        color=colors if vec_type in ['content', 'reformulation', 'related', 'implicit', 'comparative', 'entity_expansion', 'personalized'] else color,
                        symbol=symbol,
                        line=dict(width=2, color='white') if vec_type == 'head_query' else dict(width=1)
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    hovertemplate='%{text}<extra></extra>'
                )
            fig.add_trace(trace)
        
        # Calculate zoom bounds if cluster selected
        if selected_cluster != 'All' and selected_cluster_idx is not None:
            cluster_data = df[df['cluster'] == selected_cluster_idx]
            if len(cluster_data) > 0:
                # Add highlight for selected cluster with hover info
                highlight_hover_text = []
                for idx, row in cluster_data.iterrows():
                    text = f"<b>{row['label']}</b><br>"
                    text += f"Type: {row['type']}<br>"
                    text += f"Cluster: {row['cluster_name']}<br>"
                    text += f"Similarity to Head: {row['similarity_to_head']:.3f}<br>"
                    text += f"<br>Text: {row['text'][:100]}..."
                    highlight_hover_text.append(text)
                
                if projection_mode == "3D":
                    fig.add_trace(go.Scatter3d(
                        x=cluster_data['x'], 
                        y=cluster_data['y'], 
                        z=cluster_data['z'],
                        mode='markers',
                        marker=dict(
                            size=25, 
                            color='rgba(255, 0, 0, 0)', 
                            line=dict(width=3, color='red')
                        ),
                        text=highlight_hover_text,
                        hoverinfo='text',
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=False
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=cluster_data['x'], 
                        y=cluster_data['y'],
                        mode='markers',
                        marker=dict(
                            size=25, 
                            color='rgba(255, 0, 0, 0)', 
                            line=dict(width=3, color='red')
                        ),
                        text=highlight_hover_text,
                        hoverinfo='text',
                        hovertemplate='%{text}<extra></extra>',
                        showlegend=False
                    ))
        
        # Visualization boundaries (axes, grid, ticks)
        layout_args = dict(
            title="Query and Content Embedding Space",
            hovermode='closest',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
        )
        
        if projection_mode == "3D":
            layout_args['scene'] = dict(
                xaxis=dict(showgrid=True, showticklabels=True, title='X'),
                yaxis=dict(showgrid=True, showticklabels=True, title='Y'),
                zaxis=dict(showgrid=True, showticklabels=True, title='Z')
            )
            # Set zoom bounds for 3D if cluster selected
            if selected_cluster != 'All' and selected_cluster_idx is not None and len(cluster_data) > 0:
                margin = 2
                layout_args['scene']['xaxis']['range'] = [cluster_data['x'].min() - margin, cluster_data['x'].max() + margin]
                layout_args['scene']['yaxis']['range'] = [cluster_data['y'].min() - margin, cluster_data['y'].max() + margin]
                layout_args['scene']['zaxis']['range'] = [cluster_data['z'].min() - margin, cluster_data['z'].max() + margin]
        else:
            layout_args['xaxis'] = dict(showgrid=True, showticklabels=True, title='X')
            layout_args['yaxis'] = dict(showgrid=True, showticklabels=True, title='Y')
            # Set zoom bounds for 2D if cluster selected
            if selected_cluster != 'All' and selected_cluster_idx is not None and len(cluster_data) > 0:
                margin = 2
                layout_args['xaxis']['range'] = [cluster_data['x'].min() - margin, cluster_data['x'].max() + margin]
                layout_args['yaxis']['range'] = [cluster_data['y'].min() - margin, cluster_data['y'].max() + margin]
        
        fig.update_layout(**layout_args)
        st.plotly_chart(fig, use_container_width=True)
    
    # Selected vector info
    with st.expander("Selected Vector Information", expanded=False):
        st.markdown("### Inspect a Vector")
        
        # Create options for dropdown
        vector_options = []
        for idx, row in df.iterrows():
            option_text = f"{idx}: {row['label']} ({row['type']})"
            vector_options.append(option_text)
        
        # Dropdown for vector selection
        selected_option = st.selectbox(
            "Select a vector",
            options=vector_options,
            key="vector_selector"
        )
        
        # Extract index from selection
        selected_index = int(selected_option.split(":")[0])
        selected_row = df.iloc[selected_index]
        
        # Display vector information
        col1_info, col2_info = st.columns(2)
        with col1_info:
            st.markdown(f"**Type:** {selected_row['type'].replace('_', ' ').title()}")
            st.markdown(f"**Cluster:** {selected_row['cluster_name']}")
        with col2_info:
            st.markdown(f"**Similarity to Head:** {selected_row['similarity_to_head']:.3f}")
            st.markdown(f"**Index:** {selected_index}")
        
        st.markdown("**Full Text:**")
        st.info(selected_row['text'])
        
        # Show nearest neighbors
        st.markdown("**Nearest Neighbors:**")
        current_embedding = st.session_state.embeddings[selected_index]
        similarities_to_selected = cosine_similarity([current_embedding], st.session_state.embeddings)[0]
        neighbor_indices = np.argsort(similarities_to_selected)[::-1][1:4]
        
        for i, idx in enumerate(neighbor_indices, 1):
            neighbor = df.iloc[idx]
            st.markdown(f"{i}. {neighbor['label']} (similarity: {similarities_to_selected[idx]:.3f})")
    
    # Content optimization suggestions
    with st.expander("Content Optimization Suggestions", expanded=False):
        st.markdown("### Query Coverage by Content Chunks")
        query_embeddings = st.session_state.embeddings[1:len(df[df['type'] != 'content'])]
        content_embeddings = st.session_state.embeddings[len(df[df['type'] != 'content']):]
        content_labels = [row['label'] for idx, row in df.iterrows() if row['type'] == 'content']
        if len(content_embeddings) > 0:
            for i, q_emb in enumerate(query_embeddings):
                sims = cosine_similarity([q_emb], content_embeddings)[0]
                best_idx = np.argmax(sims)
                sim = sims[best_idx]
                content_label = content_labels[best_idx]
                query_label = df[df['type'] != 'content'].iloc[i+1]['label']
                if sim < 0.7:
                    st.warning(f"Query '{query_label}' is weakly covered (max sim {sim:.2f}) by content '{content_label}'")
                else:
                    st.success(f"Query '{query_label}' is well covered (sim {sim:.2f}) by content '{content_label}'")
            covered = set()
            for i, q_emb in enumerate(query_embeddings):
                sims = cosine_similarity([q_emb], content_embeddings)[0]
                best_idx = np.argmax(sims)
                covered.add(best_idx)
            uncovered = set(range(len(content_labels))) - covered
            if uncovered:
                for idx in uncovered:
                    st.info(f"Content chunk '{content_labels[idx]}' is not a top match for any query. Consider retargeting.")
        else:
            st.info("Add content chunks to see suggestions.")

    # Target Query Analysis
    with st.expander("Target Query Analysis", expanded=False):
        st.markdown("### Content to Target Query Alignment")
        
        # Get content chunks with assigned targets
        content_with_targets = []
        for chunk in st.session_state.content_chunks:
            if chunk['content'].strip() and chunk['target_query']:
                content_with_targets.append(chunk)
        
        if len(content_with_targets) > 0:
            st.markdown("Analyzing how well each content chunk aligns with its assigned target query:")
            
            for chunk in content_with_targets:
                # Find the index of the content chunk in the dataframe
                content_label = chunk['label'] or f"Chunk {chunk['id']}"
                content_row = df[df['label'] == f"Content: {content_label}"]
                
                if len(content_row) > 0:
                    content_idx = content_row.index[0]
                    
                    # Find the index of the target query
                    target_query_row = df[df['text'] == chunk['target_query']]
                    
                    if len(target_query_row) > 0:
                        target_idx = target_query_row.index[0]
                        
                        # Calculate cosine similarity
                        content_embedding = st.session_state.embeddings[content_idx]
                        target_embedding = st.session_state.embeddings[target_idx]
                        similarity = cosine_similarity([content_embedding], [target_embedding])[0][0]
                        
                        # Display results with color coding
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            st.markdown(f"**{content_label}**")
                        with col2:
                            st.markdown(f"→ *{chunk['target_query'][:50]}...*")
                        with col3:
                            if similarity >= 0.8:
                                st.success(f"{similarity:.3f}")
                            elif similarity >= 0.6:
                                st.warning(f"{similarity:.3f}")
                            else:
                                st.error(f"{similarity:.3f}")
                        
                        # Provide recommendation
                        if similarity < 0.7:
                            st.markdown(f"   💡 *Consider revising this content to better match the target query*")
                        
                        st.markdown("---")
            
            # Summary statistics
            st.markdown("### Summary")
            st.markdown("**Alignment Guidelines:**")
            st.markdown("- 🟢 **0.8+**: Excellent alignment")
            st.markdown("- 🟡 **0.6-0.8**: Good alignment, minor adjustments may help")
            st.markdown("- 🔴 **<0.6**: Poor alignment, significant revision recommended")
        else:
            st.info("Assign target queries to your content chunks to see alignment analysis.")
    
    # Full data table with download
    with st.expander("Show Full Data Table", expanded=False):
        st.markdown("### All Embeddings (Full Table)")
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="query_embedding_data.csv",
            mime="text/csv"
        )

# Info panel (right column)
with col2:
    with st.container():
        st.markdown("### Vector Information")
        if 'viz_df' in st.session_state:
            df = st.session_state.viz_df
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Vectors", len(df))
            with col_b:
                st.metric("Clusters", len(st.session_state.cluster_names))
            st.markdown("---")
            st.markdown("### Selected Vector")
            st.info("Use the 'Selected Vector Information' expander below the chart.")
        else:
            st.info("Generate a visualization to see vector information")

# Sidebar
with st.sidebar:
    st.markdown("## Quick Guide")
    st.markdown("""
    1. **Enter your head query** - The main query to analyze
    2. **Add simulated queries** - From Query Fan-Out or manually
    3. **Add content chunks** - Your content to optimize
    4. **Generate visualization** - See relationships in vector space
    5. **Analyze results** - Identify gaps and opportunities
    """)
    st.markdown("---")
    st.markdown("## Visual Guide")
    st.markdown("""
    **Shapes:**
    - 💎 Diamond = Head Query
    - ● Circle = Simulated Queries
    - ■ Square = Content Chunks

    **Colors:**
    - Each query gets a unique color
    - Content chunks use gray shades
    - Selected cluster items get red outline
    """)
    st.markdown("---")
    st.caption("Query Embedding Visualizer v2.0")