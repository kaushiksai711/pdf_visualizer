from spacy import load
import networkx as nx
from textblob import TextBlob
from app.services.pdf_processing import extract_text_from_pdf
import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

class KnowledgeGraphExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_topics_and_subtopics(self, text):
        # Split text into sentences
        sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        
        # Calculate number of topics based on text length
        n_clusters = min(max(3, len(sentences) // 20), 8)  # 1 topic per ~20 sentences, min 3, max 8
        
        # Create TF-IDF vectorizer with more features for longer texts
        max_features = min(100 + (len(sentences) // 10), 300)  # Scale features with text length
        vectorizer = TfidfVectorizer(
            max_features=max_features, 
            stop_words='english',
            ngram_range=(1, 2)
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Cluster sentences into topics
        kmeans = KMeans(n_clusters=n_clusters)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Get important terms for each cluster
        terms = vectorizer.get_feature_names_out()
        centroids = kmeans.cluster_centers_
        
        # Calculate cluster sizes for importance
        cluster_sizes = [sum(clusters == i) for i in range(n_clusters)]
        max_cluster_size = max(cluster_sizes)
        
        topics = []
        for i in range(n_clusters):
            # Scale importance based on cluster size
            importance = (cluster_sizes[i] / max_cluster_size) * 10
            
            # Get top terms for this cluster
            top_term_indices = centroids[i].argsort()[-5:][::-1]
            topic_terms = [terms[idx] for idx in top_term_indices]
            
            topics.append({
                'id': f'topic_{i}',
                'label': ' '.join(topic_terms[:2]),
                'type': 'MAIN_TOPIC',
                'sentences': [sent for j, sent in enumerate(sentences) if clusters[j] == i],
                'terms': topic_terms,
                'importance': importance
            })
        
        return topics

    def create_knowledge_graph(self, text):
        doc = self.nlp(text)
        
        # Track unique entities and concepts
        unique_nodes = {}
        edges = []
        
        # Add document root node
        root_id = 'doc_root'
        unique_nodes[root_id] = {
            'id': root_id,
            'label': 'Document',
            'type': 'DOCUMENT'
        }
        
        # Extract topics
        topics = self.extract_topics_and_subtopics(text)
        
        # Process each topic
        for topic in topics:
            topic_id = topic['id']
            if topic['label'] not in [node['label'] for node in unique_nodes.values()]:
                unique_nodes[topic_id] = {
                    'id': topic_id,
                    'label': topic['label'],
                    'type': 'MAIN_TOPIC'
                }
                
                # Connect to root
                edges.append({
                    'source': root_id,
                    'target': topic_id,
                    'relationship': 'CONTAINS'
                })
                
                # Process topic content
                topic_doc = self.nlp(' '.join(topic['sentences']))
                
                # Track entities and concepts for this topic
                topic_elements = defaultdict(int)
                
                # Extract entities
                for ent in topic_doc.ents:
                    if len(ent.text.strip()) > 2:
                        topic_elements[ent.text.lower()] += 1
                
                # Extract noun phrases
                for chunk in topic_doc.noun_chunks:
                    if len(chunk.text.strip()) > 2:
                        topic_elements[chunk.text.lower()] += 1
                
                # Scale importance based on counts
                max_entity_count = max(topic_elements.values()) if topic_elements else 1
                max_concept_count = max(topic_elements.values()) if topic_elements else 1
                
                # Add entities and concepts with scaled importance
                for ent, count in topic_elements.items():
                    importance = (count / max_entity_count) * 8  # Scale to reasonable size
                    element_id = f"element_{len(unique_nodes)}"
                    if ent not in [node['label'].lower() for node in unique_nodes.values()]:
                        unique_nodes[element_id] = {
                            'id': element_id,
                            'label': ent.title(),
                            'type': 'CONCEPT',
                            'importance': importance
                        }
                        
                        edges.append({
                            'source': topic_id,
                            'target': element_id,
                            'relationship': 'CONTAINS',
                            'weight': importance
                        })
        
        return {
            'nodes': list(unique_nodes.values()),
            'edges': edges
        }

def generate_knowledge_graph(file_path):
    """Generate knowledge graph from PDF content."""
    try:
        text = extract_text_from_pdf(file_path)
        extractor = KnowledgeGraphExtractor()
        graph = extractor.create_knowledge_graph(text)
        return graph
    except Exception as e:
        print(f"Error generating knowledge graph: {str(e)}")
        return None 