from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for plotting without a GUI
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
import io
import base64
import os
from werkzeug.utils import secure_filename
import folium
from branca.colormap import LinearColormap
import momepy

app = Flask(__name__)
# Allow requests from localhost (development) and from the Render frontend URL (production)
CORS(app, resources={r"/api/*": {"origins": ["*"]}})



ALLOWED_EXTENSIONS = {'gdb', 'shp', 'geojson', 'zip'}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Functional Class Mapping
FSYSTEM_MAP = {
    1: 'Interstate',
    2: 'Principal Arterial (Fwy/Exp)',
    3: 'Principal Arterial (Other)',
    4: 'Minor Arterial',
    5: 'Major Collector',
    6: 'Minor Collector',
    7: 'Local'
}

def get_plot_as_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return f"data:image/png;base64,{img_str}"

# Serve static pages
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/centrality')
def centrality():
    return send_from_directory('.', 'centrality.html')

@app.route('/api/analyze_osmnx', methods=['POST'])
def analyze_osmnx():

    try:
        data = request.json
        location = data.get('location')
        lat = data.get('lat')
        lon = data.get('lon')
        network_type = data.get('network_type', 'drive')

        if lat and lon:
            lat, lon = float(lat), float(lon)
            G = ox.graph_from_point((lat, lon), dist=2000, network_type=network_type)
        elif location:
            G = ox.graph_from_place(location, network_type=network_type)
        else:
            return jsonify({'error': 'Location or coordinates required'}), 400
        
        stats = calculate_network_statistics(G)

        network_description = generate_network_description(G, stats, location or f"{lat}, {lon}")

        # --- Fixed in main.py analyze_osmnx() ---
        node_gdf = ox.graph_to_gdfs(G, edges=False)
        bounds = [[node_gdf['y'].min(), node_gdf['x'].min()],
                  [node_gdf['y'].max(), node_gdf['x'].max()]]
        
        # Fixed center calculation: average of min and max for both axes
        center = [(node_gdf['y'].min() + node_gdf['y'].max()) / 2,
                  (node_gdf['x'].min() + node_gdf['x'].max()) / 2]
        
        m = folium.Map(location=center, zoom_start=14, tiles='CartoDB positron')

        edges_gdf = ox.graph_to_gdfs(G, nodes=False)

        for idx, row in edges_gdf.iterrows():
            highway = row.get('highway', 'unknown')
            if isinstance(highway, list):
                highway = highway[0]

            color = get_road_color(highway)
            weight = get_road_weight(highway)

        # Create popup with edge information
            popup_text = f"""
                <b>Road Type:</b> {highway}<br>
                <b>Length:</b> {row.get('length', 0):.2f}m<br>
                <b>Name:</b> {row.get('name', 'Unnamed')}
            """
            
            folium.PolyLine(
                locations=[(coord[1], coord[0]) for coord in row['geometry'].coords],
                color=color,
                weight=weight,
                opacity=0.7,
                popup=folium.Popup(popup_text, max_width=200)
            ).add_to(m)

        # Add nodes with different colors for intersections
        for node, node_data in G.nodes(data=True):
            degree = G.degree(node)
            if degree > 2:  # Only show intersections
                folium.CircleMarker(
                    location=[node_data['y'], node_data['x']],
                    radius=3,
                    color='#004E89',
                    fill=True,
                    fillColor='#004E89' if degree > 3 else '#1A659E',
                    fillOpacity=0.8,
                    popup=f"Intersection (degree: {degree})"
                ).add_to(m)
        
        m.fit_bounds(bounds)
        map_html = m._repr_html_()
        
        response = {
            'success': True,
            'statistics': stats,
            'network_description': network_description,
            'map_html': map_html,
            'center': center,
            'bounds': bounds
        }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


def calculate_network_statistics(G):
    """Calculate comprehensive network statistics"""
    
    # Basic stats
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    
    # Total length in km
    total_length = sum([data['length'] for u, v, data in G.edges(data=True)]) / 1000
    
    # Average degree
    degrees = dict(G.degree())
    avg_degree = sum(degrees.values()) / n_nodes if n_nodes > 0 else 0
    
    # Network density
    if n_nodes > 1:
        density = (2.0 * n_edges) / (n_nodes * (n_nodes - 1))
    else:
        density = 0
    
    # Connectivity metrics
    is_connected = nx.is_strongly_connected(G) if G.is_directed() else nx.is_connected(G)
    n_components = nx.number_strongly_connected_components(G) if G.is_directed() else nx.number_connected_components(G)
    
    # Street types distribution
    street_types = {}
    for u, v, data in G.edges(data=True):
        highway = data.get('highway', 'unknown')
        if isinstance(highway, list):
            highway = highway[0]
        street_types[highway] = street_types.get(highway, 0) + 1
    
    # Intersection analysis
    intersection_counts = {
        '3-way': 0, '4-way': 0, '5+way': 0
    }
    for node, degree in degrees.items():
        if degree == 3:
            intersection_counts['3-way'] += 1
        elif degree == 4:
            intersection_counts['4-way'] += 1
        elif degree > 4:
            intersection_counts['5+way'] += 1
    
    # Average street length
    avg_street_length = total_length / n_edges if n_edges > 0 else 0
    
    # Circuity (if possible to calculate)
    try:
        nodes_gdf = ox.graph_to_gdfs(G, edges=False)
        # Calculate approximate circuity
        avg_circuity = 1.0  # Default
    except:
        avg_circuity = 1.0
    
    return {
        'total_length': round(total_length, 2), 'total_nodes': n_nodes, 'total_edges': n_edges, 'avg_degree': round(avg_degree, 2), 
        'density': round(density, 6), 'is_connected': is_connected, 'n_components': n_components, 'street_types': street_types, 
        'intersection_counts': intersection_counts, 'avg_street_length': round(avg_street_length * 1000, 2), 'avg_circuity': round(avg_circuity, 2)
    }


def generate_network_description(G, stats, location):
    """Generate detailed network description text"""
    
    description = f"""
NETWORK ANALYSIS REPORT - OSMnx                 
╔═════════════════════════════════════════════════════════════════════════════════════════════════════╗
📍 LOCATION INFORMATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Location: {location}
Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 BASIC NETWORK STATISTICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Network Length: {stats['total_length']:.2f} km. Total Nodes (Intersections): {stats['total_nodes']:,}. 
Total Edges (Road Segments): {stats['total_edges']:,}. Average Edge Length: {stats['avg_street_length']:.2f} meters

🔗 CONNECTIVITY METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Average Node Degree: {stats['avg_degree']:.2f}. Network Density: {stats['density']:.6f}. 
Connected Network: {"Yes" if stats['is_connected'] else "No"}. Number of Components: {stats['n_components']}

🚦 INTERSECTION ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3-way Intersections: {stats['intersection_counts']['3-way']}. 4-way Intersections: {stats['intersection_counts']['4-way']}. 
Complex (5+) Intersections: {stats['intersection_counts']['5+way']}

🛣️  ROAD TYPE DISTRIBUTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""" # Add street types
    for road_type, count in sorted(stats['street_types'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_edges'] * 100) if stats['total_edges'] > 0 else 0
        description += f"{road_type.title()}: {count} ({percentage:.1f}%)\n"
    
    description += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 NETWORK INSIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Average Circuity: {stats['avg_circuity']:.2f}
• Network Efficiency: {"High" if stats['avg_degree'] > 2.5 else "Moderate" if stats['avg_degree'] > 2.0 else "Low"}
• Grid-like Structure: {"Yes" if stats['intersection_counts']['4-way'] > stats['intersection_counts']['3-way'] else "Limited"}
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
"""
    return description


def get_road_color(highway_type):
    """Get color for road based on type"""
    colors = {
        'motorway': '#E74C3C',
        'trunk': '#E67E22',
        'primary': '#F39C12',
        'secondary': '#F1C40F',
        'tertiary': '#2ECC71',
        'residential': '#3498DB',
        'service': '#95A5A6',
        'unclassified': '#BDC3C7',
        'motorway_link': '#C0392B',
        'trunk_link': '#D35400',
        'primary_link': '#D68910',
        'secondary_link': '#D4AC0D',
    }
    return colors.get(highway_type, '#7F8C8D')


def get_road_weight(highway_type):
    """Get line weight for road based on type"""
    weights = {
        'motorway': 4,
        'trunk': 3.5,
        'primary': 3,
        'secondary': 2.5,
        'tertiary': 2,
        'residential': 1.5,
        'service': 1,
        'unclassified': 1,
    }
    return weights.get(highway_type, 1)


@app.route('/api/analyze_gis', methods=['POST'])
def analyze_gis():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        filename = secure_filename(file.filename)
        
        # 1. Use a clean, absolute path to avoid /vsizip/ confusion
        # This moves the upload folder outside the project to prevent auto-reloading
        base_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.abspath(os.path.join(base_dir, '..', 'external_uploads'))
        
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)
            
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

        # 2. Normalize the path for GDAL (Forward slashes are safer on Windows for /vsizip/)
        normalized_path = os.path.abspath(filepath).replace('\\', '/')

        # 3. Load Dataset with explicit zip handling if necessary
        if filename.endswith('.zip'):
            # Use the zip:// prefix which is the geopandas/fiona standard
            gdf = gpd.read_file(f"zip://{normalized_path}", engine='pyogrio')
        else:
            gdf = gpd.read_file(normalized_path, engine='pyogrio')

        # Analysis logic aligned with Colab
        gdf = gdf.to_crs(epsg=3857)
        df = gdf.explode(index_parts=False).reset_index(drop=True)

        pcc_map = {3: 'JPCP', 5: 'CRCP'}
        if 'SURFACE_TY' in df.columns:
            df['Pavement_Category'] = df['SURFACE_TY'].map(pcc_map)
        else:
            df['Pavement_Category'] = 'Unknown'

        # Mapping and Math
        df['Functional_Class_Name'] = df['F_SYSTEM'].map(FSYSTEM_MAP)
        df['AADT_Growth_Pct'] = np.where(df['AADTRound'] > 0, 
                                       ((df['Future_AADT'] - df['AADTRound']) / df['AADTRound']) * 100, 0)
        df['K_Factor'] = df.get('K_Factor', pd.Series([0]*len(df))).fillna(0)

        hover_cols = ['THROUGH_LANES_HPMS', 'AADTRound', 'K_Factor', 'ROUTE_ID', 'Functional_Class_Name']
        if 'TC_Description' in df.columns:
            hover_cols.append('TC_Description')

        f_classes = request.form.getlist('functional_class')
        asset_status = request.form.get('asset_status')
        owner_type = request.form.get('owner_type')

        filtered_df = df.copy()

        if f_classes:
            filtered_df = filtered_df[filtered_df['F_SYSTEM'].astype(str).isin(f_classes)]

        if owner_type == 'Public':
            filtered_df = filtered_df[filtered_df['OWNERSHIP'].isin([1, 2, 3, 4])]
        elif owner_type == 'Private':
            filtered_df = filtered_df[filtered_df['OWNERSHIP'] == 26]

        

        if asset_status == 'Active' and 'STATUS' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['STATUS'].isin([1, 'A', 'Active'])]

        fig, ax = plt.subplots(figsize=(20, 12))

        if not filtered_df.empty:
            filtered_df.plot(ax=ax, color='blue', linewidth=2, alpha=0.8)
            ctx.add_basemap(ax, source=ctx.providers.Esri.WorldImagery)
            ax.set_title(f"Filtered Network View ({', '.join(f_classes)})")

        else:
            # Fallback if no data matches filters
            ax.text(0.5, 0.5, 'No data matches selected filters', 
                    horizontalalignment='center', verticalalignment='center', fontsize=20)


        ax.set_axis_off()
        map_img = get_plot_as_base64()

        inventory_cols = [
            'ROUTE_ID', 'F_SYSTEM', 'SURFACE_INC', 'Mileage', 
            'SHOULDER_WIDTH_R', 'THROUGH_LANES_HPMS', 'isPaved', 
            'AADTRound', 'YEAR_BUILT', 'SPEED_LIMIT', 'GRADIENT', 
            'CRACKING', 'YEAR_LAST_IMPROVED', 'THICKNESS'
        ]

        # Check which columns actually exist in the dataframe to avoid KeyErrors
        available_cols = [c for c in inventory_cols if c in filtered_df.columns]

        return jsonify({
            'success': True,
            'map_img': map_img,
            'asset_inventory': filtered_df[available_cols].head(100).to_dict(orient='records'),
            'stats': {
                'county_stats': filtered_df.groupby('COUNTY_ID')[['VMT', 'TruckVMT']].sum().reset_index().to_dict(orient='records'),
                'lane_utilization': filtered_df[hover_cols].dropna().sample(min(800, len(filtered_df))).to_dict(orient='records'),
                'vmt_by_class': filtered_df.groupby('Functional_Class_Name')['VMT'].sum().reset_index().to_dict(orient='records'),
                'growth_stats': filtered_df.groupby('Functional_Class_Name')['AADT_Growth_Pct'].mean().reset_index().to_dict(orient='records'),
                'mileage_stats': filtered_df.groupby('Functional_Class_Name')['Mileage'].sum().reset_index().to_dict(orient='records'),
            }
            
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    

@app.route('/api/analyze_centrality', methods=['POST'])
def analyze_centrality():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        top_n = int(request.form.get('top_n', 10))
        centrality_type = request.form.get('centrality_type', 'edge_betweenness')
        filename = secure_filename(file.filename)
        
        # 1. Use the EXACT same absolute path logic from analyze_gis
        base_dir = os.path.dirname(os.path.abspath(__file__))
        upload_dir = os.path.normpath(os.path.join(base_dir, '..', 'external_uploads'))
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir, exist_ok=True)
   
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)


        # 2. Normalize the path (Crucial for Windows/GDAL compatibility)
        normalized_path = os.path.abspath(filepath).replace('\\', '/')
        if filename.endswith('.zip'):
            gdf = gpd.read_file(f"zip://{normalized_path}", engine='pyogrio')
        else:
            gdf = gpd.read_file(normalized_path, engine='pyogrio')


        # 3. Geometry Cleaning:
        # Step 1: Project tp a planar CRS for momepy weights
        gdf = gdf.to_crs(epsg=26917)
        
        # 2. Geometry Cleaning
        gdf_exploded = gdf.explode(index_parts=False).reset_index(drop=True)
        gdf_clean = gdf_exploded[gdf_exploded.geometry.type == 'LineString'].copy()

        # 2. Build Topology using momepy
        primal = momepy.gdf_to_nx(gdf_clean, approach='primal')

        # 3. Calculation Logic (The Bridge between momepy and networkx)
        # column_name must match what you use in plotting and statistics
        centrality_type = request.form.get('centrality_type', 'betweenness_node')
        column_name = 'centrality_score'
        is_edge_based = "edge" in centrality_type

        if centrality_type == 'closeness':
            # Momepy still wraps some weighted metrics, but networkx is safer
            scores = nx.closeness_centrality(primal, distance='mm_len')
            nx.set_node_attributes(primal, scores, column_name)
            
        elif centrality_type == 'betweenness_node':
            scores = nx.betweenness_centrality(primal, k=min(1000, len(primal)), weight='mm_len')
            nx.set_node_attributes(primal, scores, column_name)
            
        elif centrality_type == 'betweenness_edge':
            # Edge betweenness returns a dictionary keyed by (u, v)
            scores = nx.edge_betweenness_centrality(primal, k=min(1000, len(primal)), weight='mm_len')
            nx.set_edge_attributes(primal, scores, column_name)
            
        elif centrality_type == 'eigenvector':
            scores = nx.eigenvector_centrality(primal, max_iter=1000, weight='mm_len')
            nx.set_node_attributes(primal, scores, column_name)
            
        else: # Degree
            scores = nx.degree_centrality(primal)
            nx.set_node_attributes(primal, scores, column_name)

        # 4. Convert back to GeoDataFrames
        nodes, edges = momepy.nx_to_gdf(primal)

        # 5. Visualization Logic (The Heatmap Fix)
        fig, ax = plt.subplots(figsize=(12, 12))
        
        if is_edge_based:
            # Heatmap on roads, faint dots for intersections
            nodes.plot(ax=ax, color='#cccccc', markersize=5, alpha=0.3, zorder=1)
            edges.plot(ax=ax, column=column_name, cmap='viridis', linewidth=3, legend=True, zorder=2)
            top_data = edges.sort_values(by=column_name, ascending=False).head(top_n)
        else:
            # Heatmap on intersections, faint lines for roads
            edges.plot(ax=ax, color='#dee2e6', linewidth=1, alpha=0.5, zorder=1)
            nodes.plot(ax=ax, column=column_name, cmap='magma', markersize=50, legend=True, zorder=2)
            top_data = nodes.sort_values(by=column_name, ascending=False).head(top_n)

        # Highlight top N critical elements in Red
        top_data.plot(ax=ax, color='red', markersize=100, linewidth=5, edgecolors='white', zorder=3)
        
        ax.set_axis_off()
        map_img = get_plot_as_base64()

        # 6. Response Construction
        stats_gdf = edges if is_edge_based else nodes
        top_elements = []
        for _, row in top_data.iterrows():
            label = f"Edge {row.get('id', 'N/A')}" if is_edge_based else f"Node ({round(row.geometry.x,1)}, {round(row.geometry.y,1)})"
            top_elements.append({"id": label, "centrality": float(row[column_name])})

        return jsonify({
            'success': True,
            'map_img': map_img,
            'network_info': {'total_nodes': len(nodes), 'total_edges': len(edges)},
            'statistics': {
                'max': float(stats_gdf[column_name].max()),
                'min': float(stats_gdf[column_name].min()),
                'mean': float(stats_gdf[column_name].mean())
            },
            'top_elements': top_elements,
            'distribution': {
                'bins': [float(b) for b in np.histogram(stats_gdf[column_name].fillna(0), bins=10)[1]],
                'frequencies': [int(f) for f in np.histogram(stats_gdf[column_name].fillna(0), bins=10)[0]]
            }
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host='0.0.0.0', port=port, debug=debug_mode, use_reloader=False)