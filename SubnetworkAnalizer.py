import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import networkx as nx
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class SubnetworkAnalyzer:
    def __init__(self, df_edges):
        """
        Initialize the analyzer with edge dataframe
        df_edges should have columns: source, target, relation_type, weight
        """
        self.df_edges = df_edges
        self.G = None
        self.components = None
        self.component_stats = None
        
    def build_graph(self):
        """Build igraph from edge dataframe"""
        edges = list(zip(self.df_edges["source"], self.df_edges["target"]))
        self.G = ig.Graph(directed=True)
        
        # Get all unique nodes
        all_nodes = list(set(self.df_edges["source"]) | set(self.df_edges["target"]))
        self.G.add_vertices(all_nodes)
        self.G.add_edges(edges)
        
        # Add edge attributes
        self.G.es["relation_type"] = self.df_edges["relation_type"].tolist()
        self.G.es["weight"] = self.df_edges["weight"].tolist()
        
        print(f"Graph built with {self.G.vcount()} vertices and {self.G.ecount()} edges")
        return self.G
    
    def identify_components(self, mode='weak'):
        """
        Identify connected components
        mode: 'weak' for weakly connected, 'strong' for strongly connected
        """
        if self.G is None:
            self.build_graph()
            
        if mode == 'weak':
            self.components = self.G.components(mode='weak')
        else:
            self.components = self.G.components(mode='strong')
            
        print(f"Found {len(self.components)} {mode}ly connected components")
        
        # Calculate component statistics
        component_sizes = [len(comp) for comp in self.components]
        self.component_stats = {
            'num_components': len(self.components),
            'sizes': component_sizes,
            'largest_size': max(component_sizes),
            'smallest_size': min(component_sizes),
            'avg_size': np.mean(component_sizes),
            'median_size': np.median(component_sizes)
        }
        
        return self.components
    
    def analyze_component_isolation(self, min_size=2):
        """
        Analyze how isolated each component is from others
        """
        if self.components is None:
            self.identify_components()
            
        isolation_scores = []
        component_details = []
        
        for i, component in enumerate(self.components):
            if len(component) < min_size:
                continue
                
            # Create subgraph for this component
            subgraph = self.G.subgraph(component)
            
            # Calculate internal metrics
            internal_edges = subgraph.ecount()
            internal_density = subgraph.density()
            
            # Calculate strength/degree metrics
            vertex_strength = subgraph.strength(weights=subgraph.es["weight"])
            avg_strength = np.mean(vertex_strength)
            
            # Analyze relation types within component
            relation_counts = Counter(subgraph.es["relation_type"])
            dominant_relation = relation_counts.most_common(1)[0] if relation_counts else ("None", 0)
            
            # Calculate clustering coefficient (undirected version)
            try:
                undirected_sub = subgraph.as_undirected()
                avg_clustering = undirected_sub.transitivity_avglocal_undirected()
            except:
                avg_clustering = 0
            
            component_info = {
                'component_id': i,
                'size': len(component),
                'internal_edges': internal_edges,
                'density': internal_density,
                'avg_strength': avg_strength,
                'avg_clustering': avg_clustering,
                'dominant_relation': dominant_relation[0],
                'dominant_relation_count': dominant_relation[1],
                'relation_diversity': len(relation_counts),
                'nodes': [self.G.vs[v]["name"] for v in component]
            }
            
            component_details.append(component_info)
            
        return component_details
    
    def find_bridge_nodes(self):
        """
        Find nodes that connect different components (bridge nodes)
        """
        if self.G is None:
            self.build_graph()
            
        # Convert to undirected for bridge analysis
        G_undirected = self.G.as_undirected()
        
        # Find articulation points (nodes whose removal increases components)
        articulation_points = []
        original_components = len(G_undirected.components())
        
        for v in range(G_undirected.vcount()):
            # Remove vertex and check component count
            G_temp = G_undirected.copy()
            G_temp.delete_vertices([v])
            new_components = len(G_temp.components())
            
            if new_components > original_components:
                articulation_points.append({
                    'node': self.G.vs[v]["name"],
                    'original_components': original_components,
                    'new_components': new_components,
                    'strength': self.G.strength(v, weights=self.G.es["weight"])
                })
        
        return articulation_points
    
    def analyze_inter_component_connections(self):
        """
        Analyze connections between different components
        """
        if self.components is None:
            self.identify_components()
            
        # Create node to component mapping
        node_to_component = {}
        for comp_id, component in enumerate(self.components):
            for node in component:
                node_to_component[node] = comp_id
        
        # Find inter-component edges
        inter_component_edges = []
        for edge in self.G.es:
            source_comp = node_to_component[edge.source]
            target_comp = node_to_component[edge.target]
            
            if source_comp != target_comp:
                inter_component_edges.append({
                    'source': self.G.vs[edge.source]["name"],
                    'target': self.G.vs[edge.target]["name"],
                    'source_component': source_comp,
                    'target_component': target_comp,
                    'relation_type': edge["relation_type"],
                    'weight': edge["weight"]
                })
        
        return inter_component_edges
    
    def detect_community_structure(self, algorithm='leiden'):
        """
        Detect communities using various algorithms - safer version
        """
        if self.G is None:
            self.build_graph()
        
        # Create undirected graph more carefully
        # Method 1: Use combine_edges parameter
        try:
            G_undirected = self.G.as_undirected(combine_edges="sum")
            print("Successfully converted to undirected using combine_edges='sum'")
        except:
            try:
                # Method 2: Manual conversion
                print("Fallback: Manual undirected conversion")
                edges_undirected = []
                weights_undirected = []
                relation_types_undirected = []
                
                # Create edge dictionary to handle duplicates
                edge_dict = defaultdict(list)
                
                for edge in self.G.es:
                    source = edge.source
                    target = edge.target
                    weight = edge["weight"]
                    relation = edge["relation_type"]
                    
                    # Create undirected edge key (always smaller index first)
                    key = tuple(sorted([source, target]))
                    edge_dict[key].append((weight, relation))
                
                # Create undirected edges with combined weights
                for (v1, v2), edge_data in edge_dict.items():
                    edges_undirected.append((v1, v2))
                    # Sum weights for multiple edges
                    total_weight = sum([w for w, r in edge_data])
                    weights_undirected.append(total_weight)
                    # Use most common relation type
                    relations = [r for w, r in edge_data]
                    most_common_relation = Counter(relations).most_common(1)[0][0]
                    relation_types_undirected.append(most_common_relation)
                
                # Create new undirected graph
                G_undirected = ig.Graph(directed=False)
                G_undirected.add_vertices(self.G.vcount())
                
                # Copy vertex attributes
                if "name" in self.G.vs.attributes():
                    G_undirected.vs["name"] = self.G.vs["name"]
                
                # Add edges and attributes
                G_undirected.add_edges(edges_undirected)
                G_undirected.es["weight"] = weights_undirected
                G_undirected.es["relation_type"] = relation_types_undirected
                
            except Exception as e:
                print(f"Manual conversion also failed: {str(e)}")
                # Last resort: create simple undirected without attributes
                G_undirected = self.G.as_undirected()
                G_undirected.es["weight"] = [1.0] * G_undirected.ecount()
        
        # Ensure weights are float (some algorithms are picky about this)
        try:
            G_undirected.es["weight"] = [float(w) for w in G_undirected.es["weight"]]
        except:
            G_undirected.es["weight"] = [1.0] * G_undirected.ecount()
        
        # Community detection with error handling
        try:
            if algorithm == 'leiden':
                communities = G_undirected.community_leiden(weights=G_undirected.es["weight"])
            elif algorithm == 'louvain':
                communities = G_undirected.community_multilevel(weights=G_undirected.es["weight"])
            elif algorithm == 'infomap':
                communities = G_undirected.community_infomap(edge_weights=G_undirected.es["weight"])
            else:
                communities = G_undirected.community_fastgreedy(weights=G_undirected.es["weight"]).as_clustering()
            
        except Exception as e:
            print(f"Weighted community detection failed: {str(e)}")
            print("Trying unweighted community detection...")
            
            # Try without weights
            if algorithm == 'leiden':
                communities = G_undirected.community_leiden()
            elif algorithm == 'louvain':
                communities = G_undirected.community_multilevel()
            elif algorithm == 'infomap':
                communities = G_undirected.community_infomap()
            else:
                communities = G_undirected.community_fastgreedy().as_clustering()
        
        # Calculate modularity safely
        try:
            modularity_score = communities.modularity
        except:
            modularity_score = 0.0
        
        # Analyze communities
        community_analysis = []
        for i, community in enumerate(communities):
            try:
                subgraph = G_undirected.subgraph(community)
                
                # Calculate community metrics
                internal_edges = subgraph.ecount()
                size = len(community)
                density = subgraph.density() if size > 1 else 0.0
                
                # Analyze relation types in community (use original directed graph)
                community_edges = []
                for e in self.G.es:
                    if e.source in community and e.target in community:
                        community_edges.append(e)
                
                relation_types = [e["relation_type"] for e in community_edges]
                relation_counter = Counter(relation_types)
                
                # Get node names safely
                node_names = []
                for v in community:
                    try:
                        if hasattr(self.G.vs[v], '__getitem__') and "name" in self.G.vs.attributes():
                            node_names.append(self.G.vs[v]["name"])
                        else:
                            node_names.append(str(v))
                    except:
                        node_names.append(str(v))
                
                community_info = {
                    'community_id': i,
                    'size': size,
                    'density': density,
                    'internal_edges': internal_edges,
                    'modularity': modularity_score,
                    'nodes': node_names,
                    'relation_types': dict(relation_counter),
                    'dominant_relation': relation_counter.most_common(1)[0] if relation_counter else None
                }
                
                community_analysis.append(community_info)
                
            except Exception as e:
                print(f"Error analyzing community {i}: {str(e)}")
                continue
        
        return community_analysis, modularity_score
    
    def generate_report(self, min_component_size=2, output_file=None):
        """
        Generate comprehensive analysis report
        """
        report = []
        report.append("="*60)
        report.append("SUBNETWORK ANALYSIS REPORT")
        report.append("="*60)
        
        # Build graph if not already built
        if self.G is None:
            self.build_graph()
        
        # Basic graph statistics
        report.append(f"\nGRAPH OVERVIEW:")
        report.append(f"- Total nodes: {self.G.vcount()}")
        report.append(f"- Total edges: {self.G.ecount()}")
        report.append(f"- Graph density: {self.G.density():.4f}")
        report.append(f"- Is directed: {self.G.is_directed()}")
        
        # Component analysis
        components = self.identify_components()
        report.append(f"\nCONNECTED COMPONENTS:")
        report.append(f"- Number of components: {self.component_stats['num_components']}")
        report.append(f"- Largest component size: {self.component_stats['largest_size']}")
        report.append(f"- Average component size: {self.component_stats['avg_size']:.2f}")
        
        # Detailed component analysis
        component_details = self.analyze_component_isolation(min_component_size)
        report.append(f"\nDETAILED COMPONENT ANALYSIS (size >= {min_component_size}):")
        
        for comp in sorted(component_details, key=lambda x: x['size'], reverse=True)[:10]:  # Top 10
            report.append(f"\nComponent {comp['component_id']}:")
            report.append(f"  - Size: {comp['size']} nodes")
            report.append(f"  - Internal edges: {comp['internal_edges']}")
            report.append(f"  - Density: {comp['density']:.4f}")
            report.append(f"  - Average strength: {comp['avg_strength']:.2f}")
            report.append(f"  - Clustering coefficient: {comp['avg_clustering']:.4f}")
            report.append(f"  - Dominant relation: {comp['dominant_relation']} ({comp['dominant_relation_count']} edges)")
            report.append(f"  - Relation diversity: {comp['relation_diversity']} types")
            if comp['size'] <= 10:  # Show nodes for small components
                report.append(f"  - Nodes: {', '.join(comp['nodes'])}")
        
        # Bridge nodes analysis
        bridge_nodes = self.find_bridge_nodes()
        if bridge_nodes:
            report.append(f"\nBRIDGE NODES (Critical for connectivity):")
            for bridge in bridge_nodes[:5]:  # Top 5
                report.append(f"  - {bridge['node']}: strength={bridge['strength']:.2f}")
        
        # Inter-component connections
        inter_edges = self.analyze_inter_component_connections()
        if inter_edges:
            report.append(f"\nINTER-COMPONENT CONNECTIONS: {len(inter_edges)} edges")
            relation_types = Counter([e['relation_type'] for e in inter_edges])
            report.append(f"  - Most common inter-component relations:")
            for rel_type, count in relation_types.most_common(5):
                report.append(f"    * {rel_type}: {count} edges")
        
        # Community detection
        try:
            communities, modularity = self.detect_community_structure()
            report.append(f"\nCOMMUNITY STRUCTURE (Leiden algorithm):")
            report.append(f"- Number of communities: {len(communities)}")
            report.append(f"- Modularity score: {modularity:.4f}")
            
            # Top communities by size
            top_communities = sorted(communities, key=lambda x: x['size'], reverse=True)[:5]
            for comm in top_communities:
                report.append(f"\nCommunity {comm['community_id']}:")
                report.append(f"  - Size: {comm['size']} nodes")
                report.append(f"  - Density: {comm['density']:.4f}")
                report.append(f"  - Dominant relation: {comm['dominant_relation']}")
        except Exception as e:
            report.append(f"\nCommunity detection failed: {str(e)}")
        
        report.append("\n" + "="*60)
        
        report_text = "\n".join(report)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to {output_file}")
        
        return report_text
    
    def visualize_component_distribution(self):
        """
        Create visualizations of component analysis
        """
        if self.components is None:
            self.identify_components()
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Component size distribution
        sizes = [len(comp) for comp in self.components]
        axes[0,0].hist(sizes, bins=min(20, len(set(sizes))), edgecolor='black')
        axes[0,0].set_title('Component Size Distribution')
        axes[0,0].set_xlabel('Component Size')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_yscale('log')
        
        # Cumulative size distribution
        sorted_sizes = sorted(sizes, reverse=True)
        cumulative_nodes = np.cumsum(sorted_sizes)
        axes[0,1].plot(range(1, len(sorted_sizes)+1), cumulative_nodes)
        axes[0,1].set_title('Cumulative Nodes by Component Rank')
        axes[0,1].set_xlabel('Component Rank')
        axes[0,1].set_ylabel('Cumulative Nodes')
        
        # Relation type distribution across components
        component_details = self.analyze_component_isolation(min_size=1)
        relation_data = []
        for comp in component_details:
            if comp['size'] >= 2:  # Only components with edges
                relation_data.append(comp['relation_diversity'])
        
        if relation_data:
            axes[1,0].hist(relation_data, bins=max(1, len(set(relation_data))), edgecolor='black')
            axes[1,0].set_title('Relation Type Diversity per Component')
            axes[1,0].set_xlabel('Number of Relation Types')
            axes[1,0].set_ylabel('Number of Components')
        
        # Component density vs size
        densities = [comp['density'] for comp in component_details if comp['size'] >= 2]
        comp_sizes = [comp['size'] for comp in component_details if comp['size'] >= 2]
        
        if densities and comp_sizes:
            axes[1,1].scatter(comp_sizes, densities, alpha=0.6)
            axes[1,1].set_title('Component Density vs Size')
            axes[1,1].set_xlabel('Component Size')
            axes[1,1].set_ylabel('Density')
            axes[1,1].set_xscale('log')
        
        plt.tight_layout()
        plt.show()
        
        return fig