from spacy import load
import networkx as nx
from textblob import TextBlob
from app.services.pdf_processing import extract_text_from_pdf
import spacy
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import pytextrank

import re
import numpy as np
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import pytextrank

class EnhancedKnowledgeGraphExtractor:
    def __init__(self):
        # Lazy loading of models to improve initial startup time
        self._nlp = None
        self._sentence_model = None
        self._keybert_model = None
        
    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = spacy.load("en_core_web_sm")
            self._nlp.add_pipe("textrank")
        return self._nlp
    
    @property
    def sentence_model(self):
        if self._sentence_model is None:
            self._sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._sentence_model
    
    @property
    def keybert_model(self):
        if self._keybert_model is None:
            self._keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
        return self._keybert_model
        
    def preprocess_text(self, text):
        """Optimized text preprocessing"""
        # Use faster regex replacements
        text = re.sub(r'\s+', ' ', text)  # Whitespace normalization
        text = re.sub(r'\[\d+\]', '', text)  # Remove references
        
        # Combine character removal in one pass
        text = re.sub(r'[\u2022\u2023\u2043\u204C\u204D\u2219\u25D8\u25E6\u2619\u2765\u2767\-]', ' ', text)
        
        # Use spaCy's sentence segmentation more efficiently
        doc = self.nlp(text)
        sentences = [
            re.sub(r'[^\w\s\.,:;\(\)\/]', ' ', sent.text.strip())
            for sent in doc.sents 
            if len(sent.text.strip()) > 20
        ]
        
        # Use a generator for memory efficiency
        return [
            re.sub(r'\s+', ' ', sent).strip() 
            for sent in sentences 
            if len(sent) > 20
        ]
        
    def extract_topics_and_subtopics(self, text):
        # Preprocess and split text into sentences
        sentences = list(self.preprocess_text(text))
        
        if len(sentences) < 10:
            return [{
                'id': 'topic_0',
                'label': 'Main Content',
                'type': 'MAIN_TOPIC',
                'sentences': sentences,
                'terms': [],
                'importance': 10
            }]
        
        # Adaptive clustering with more efficient parameters
        n_clusters = min(max(3, len(sentences) // 15), 10)
        
        # Optimized TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            max_features=300, 
            stop_words='english',
            ngram_range=(1, 3),
            min_df=2
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # More efficient embedding computation
        embeddings = self.sentence_model.encode(sentences, batch_size=32, show_progress_bar=False)
        
        # Combine features with NumPy operations
        combined_features = np.hstack((
            0.4 * tfidf_matrix.toarray(),
            0.6 * embeddings
        ))
        
        # Use more efficient clustering
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10,  # More stable initialization
            init='k-means++',  # Better initial centroid selection
        )
        clusters = kmeans.fit_predict(combined_features)
        
        terms = vectorizer.get_feature_names_out()
        
        topics = []
        for i in range(n_clusters):
            cluster_sentences = [sent for j, sent in enumerate(sentences) if clusters[j] == i]
            
            if not cluster_sentences:
                continue
                
            cluster_indices = [j for j, c in enumerate(clusters) if c == i]
            
            # Vectorized computations
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
            cluster_tfidf = np.asarray(cluster_tfidf).flatten()
            
            # Efficient keyword extraction
            cluster_text = " ".join(cluster_sentences)
            keyphrases = self.keybert_model.extract_keywords(
                cluster_text, 
                keyphrase_ngram_range=(1, 3), 
                stop_words='english', 
                use_mmr=True,
                diversity=0.7,
                top_n=15    
            )
        
            # Use set for faster uniqueness check
            topic_terms = []
            seen_terms = set()
            for term, _ in keyphrases:
                clean_term = re.sub(r'[^\w\s\-]', '', term).strip()
                if (len(clean_term) >= 3 and 
                    re.search(r'[a-zA-Z]', clean_term) and 
                    clean_term not in seen_terms):
                    topic_terms.append(clean_term)
                    seen_terms.add(clean_term)
                    if len(topic_terms) >= 10:
                        break
            
            # Efficient central sentence selection
            cluster_embeddings = embeddings[cluster_indices]
            similarities = cosine_similarity(cluster_embeddings)
            central_sentence_idx = cluster_indices[np.argmax(similarities.mean(axis=1))]
            
            representative_sentence = sentences[central_sentence_idx]
            
            # TextRank phrase extraction
            doc = self.nlp(representative_sentence)
            phrases = [
                phrase.text for phrase in doc._.phrases[:3] 
                if len(phrase.text) >= 3
            ]
            
            # Smart label generation
            label = phrases[0] if phrases else (
                next((term for term in topic_terms if ' ' in term), 
                     topic_terms[0] if topic_terms else f"Topic {i+1}")
            )
            
            # Importance calculation
            cluster_size = len(cluster_sentences)
            coherence = similarities.mean()
            importance = (cluster_size / len(sentences) * 7) + (coherence * 3)
            
            topics.append({
                'id': f'topic_{i}',
                'label': label,
                'type': 'MAIN_TOPIC',
                'sentences': cluster_sentences,
                'terms': topic_terms[:10],
                'importance': importance,
                'representative_sentence': representative_sentence
            })
        
        return topics
    
    def extract_entities_and_concepts(self, sentences):
        """Extract entities and concepts from text with improved processing"""
        combined_text = " ".join(sentences)
        doc = self.nlp(combined_text)
        
        entities = {}
        concepts = {}
        domain_specific_terms = {}
        
        # Use KeyBERT for improved terminology extraction
        keyphrases = self.keybert_model.extract_keywords(
            combined_text, 
            keyphrase_ngram_range=(1, 4),  # Increased from (1,3) to catch longer domain terms
            stop_words='english', 
            use_mmr=True,
            diversity=0.6,  # Slightly decreased to allow more related terms
            top_n=30  # Increased from default to capture more candidates
        )
        
        # Add KeyBERT extracted terms to domain_specific_terms
        for term, score in keyphrases:
            if len(term) > 3 and re.search(r'[a-zA-Z]', term):
                clean_term = re.sub(r'[^\w\s\-]', '', term).strip()
                if len(clean_term) > 3:
                    key = clean_term.lower()
                    if key not in domain_specific_terms:
                        # Find contexts for the term
                        contexts = []
                        for sent in sentences:
                            if re.search(r'\b' + re.escape(key) + r'\b', sent.lower()):
                                contexts.append(sent)
                        
                        domain_specific_terms[key] = {
                            'text': clean_term,
                            'count': sum(1 for sent in sentences if re.search(r'\b' + re.escape(key) + r'\b', sent.lower())),
                            'rank': score * 2,  # Weight KeyBERT scores more heavily
                            'mentions': contexts[:2]  # Store up to 2 contexts
                        }
        
        # Process named entities
        for ent in doc.ents:
            text = ent.text.strip()
            if len(text) > 2:
                # Clean the entity text
                text = re.sub(r'[^\w\s\-]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if not text or len(text) < 3 or not re.search(r'[a-zA-Z]', text):
                    continue
                
                key = text.lower()
                
                if key not in entities:
                    entities[key] = {
                        'text': text,
                        'type': ent.label_,
                        'count': 0,
                        'mentions': [],
                        'score': 0
                    }
                    
                entities[key]['count'] += 1
                context = doc[max(0, ent.start - 5):min(len(doc), ent.end + 5)].text
                # Clean the context
                context = re.sub(r'[^\w\s\-\.,;:\(\)]', ' ', context)
                context = re.sub(r'\s+', ' ', context).strip()
                entities[key]['mentions'].append(context)
        
        # Process noun phrases for concepts with improved filtering
        for chunk in doc.noun_chunks:
            text = chunk.text.strip()
            if len(text) > 3:
                # Clean the concept text
                text = re.sub(r'[^\w\s\-]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if not text or len(text) < 3 or not re.search(r'[a-zA-Z]', text):
                    continue
                    
                # Filter out common, non-specific phrases
                if text.lower() in ['the one', 'this one', 'that one', 'many', 'some', 'few', 'the way', 'the time']:
                    continue
                    
                # Check if it contains at least one noun
                chunk_doc = self.nlp(text)
                if not any(token.pos_ in ["NOUN", "PROPN"] for token in chunk_doc):
                    continue
                
                key = text.lower()
                
                # Check if this is not already tracked as an entity
                if key not in entities and key not in concepts:
                    concepts[key] = {
                        'text': text,
                        'count': 0,
                        'mentions': [],
                        'score': 0
                    }
                
                if key in concepts:
                    concepts[key]['count'] += 1
                    context = doc[max(0, chunk.start - 5):min(len(doc), chunk.end + 5)].text
                    # Clean the context
                    context = re.sub(r'[^\w\s\-\.,;:\(\)]', ' ', context)
                    context = re.sub(r'\s+', ' ', context).strip()
                    concepts[key]['mentions'].append(context)
        
        # Process TextRank phrases
        for phrase in doc._.phrases[:40]:  # Increased from default to get more phrases
            text = phrase.text.strip()
            if len(text) > 3:
                # Clean the phrase
                text = re.sub(r'[^\w\s\-]', '', text)
                text = re.sub(r'\s+', ' ', text).strip()
                
                if not text or len(text) < 3 or not re.search(r'[a-zA-Z]', text):
                    continue
                
                key = text.lower()
                
                # Check if this is not already tracked as an entity or concept
                if key not in entities and key not in concepts and key not in domain_specific_terms:
                    domain_specific_terms[key] = {
                        'text': text,
                        'count': 1,
                        'rank': phrase.rank,
                        'mentions': []
                    }
                
                    # Get text context
                    contexts = []
                    for sent in sentences:
                        if re.search(r'\b' + re.escape(key) + r'\b', sent.lower()):
                            contexts.append(sent)
                    
                    if contexts:
                        domain_specific_terms[key]['mentions'] = contexts[:2]
                else:
                    if key in domain_specific_terms:
                        domain_specific_terms[key]['count'] += 1
        
        # Enhanced scoring that combines frequency, position and semantic importance
        # Calculate scores for entities
        for key in entities:
            # Base score from count
            entities[key]['score'] = entities[key]['count'] * 1.2
            
            # Boost score if the entity appears in the TextRank phrases
            for phrase in doc._.phrases[:20]:
                if key in phrase.text.lower():
                    entities[key]['score'] += phrase.rank * 5
                    
            # Additional boost for entities that appear in the first 20% of sentences
            early_sentences = sentences[:max(1, len(sentences) // 5)]
            early_text = " ".join(early_sentences).lower()
            if key in early_text:
                entities[key]['score'] += 2
        
        # Calculate scores for concepts
        for key in concepts:
            # Base score from count
            concepts[key]['score'] = concepts[key]['count']
            
            # Boost score if the concept appears in the TextRank phrases
            for phrase in doc._.phrases[:20]:
                if key in phrase.text.lower():
                    concepts[key]['score'] += phrase.rank * 5
                    
            # Additional boost for concepts that appear in KeyBERT results
            if key in [k for k, _ in keyphrases]:
                concepts[key]['score'] += 3
                
            # Look for multi-word concepts (generally more specific)
            if len(key.split()) > 1:
                concepts[key]['score'] += 1
        
        # Add TF-IDF scores to improve domain term relevance
        if len(sentences) >= 5:  # Only if we have enough sentences for meaningful TF-IDF
            try:
                # Create a TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_features=300,
                    stop_words='english',
                    ngram_range=(1, 3)
                )
                tfidf_matrix = vectorizer.fit_transform(sentences)
                feature_names = vectorizer.get_feature_names_out()
                
                # Get average TF-IDF scores for each term
                avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
                
                # Map scores to terms
                term_tfidf = {feature_names[i]: avg_tfidf[i] for i in range(len(feature_names))}
                
                # Boost domain terms based on TF-IDF scores
                for key in domain_specific_terms:
                    # Look for exact matches or contained terms
                    matched_terms = [t for t in term_tfidf if t in key or key in t]
                    if matched_terms:
                        tfidf_boost = sum(term_tfidf[t] for t in matched_terms) * 5
                        domain_specific_terms[key]['rank'] += tfidf_boost
            except:
                # Fallback if TF-IDF calculation fails
                pass
        
        # Filter domain terms to keep only the most relevant
        filtered_domain_terms = {}
        for key, term in sorted(domain_specific_terms.items(), key=lambda x: x[1]['rank'], reverse=True):
            if len(filtered_domain_terms) >= 50:  # Get top 50 terms
                break
                
            # Apply stricter filtering for quality
            if term['count'] >= 1 and term['rank'] > 0.02:  # Increased rank threshold
                # Check if term is fully contained in a longer, higher-ranked term
                is_subterm = False
                for existing_key in filtered_domain_terms:
                    if key in existing_key and key != existing_key and filtered_domain_terms[existing_key]['rank'] > term['rank']:
                        is_subterm = True
                        break
                        
                if not is_subterm:
                    filtered_domain_terms[key] = term
        
        return entities, concepts, filtered_domain_terms

    def extract_relationships(self, sentences):
        """Extract semantic relationships between entities"""
        relationships = []
        
        for sentence in sentences:
            # Clean the sentence to avoid parsing issues
            clean_sentence = re.sub(r'[^\w\s\-\.,;:\(\)]', ' ', sentence)
            clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
            
            doc = self.nlp(clean_sentence)
            
            # Extract subject-verb-object patterns
            for sent in doc.sents:
                entities = [(e.text, e.start, e.label_) for e in sent.ents]
                if len(entities) < 2:
                    continue
                    
                # Find verb-mediated relationships
                for token in sent:
                    if token.dep_ == "ROOT" and token.pos_ == "VERB":
                        subject = None
                        direct_object = None
                        
                        # Find subject
                        for child in token.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                subject = self._get_span_with_compounds(child)
                                break
                        
                        # Find object
                        for child in token.children:
                            if child.dep_ in ["dobj", "pobj"]:
                                direct_object = self._get_span_with_compounds(child)
                                break
                        
                        if subject and direct_object:
                            # Clean subject and object
                            subject = re.sub(r'[^\w\s\-]', ' ', subject).strip()
                            direct_object = re.sub(r'[^\w\s\-]', ' ', direct_object).strip()
                            
                            if len(subject) > 2 and len(direct_object) > 2:
                                relationships.append({
                                    'source': subject,
                                    'target': direct_object,
                                    'relationship': token.lemma_,
                                    'confidence': 0.7  # Base confidence score
                                })
                                
                # Also extract prepositional relationships (X of Y, X for Y, etc.)
                for token in sent:
                    if token.pos_ == "NOUN" and any(child.pos_ == "ADP" for child in token.children):
                        for child in token.children:
                            if child.pos_ == "ADP" and any(grandchild.pos_ in ["NOUN", "PROPN"] for grandchild in child.children):
                                for grandchild in child.children:
                                    if grandchild.pos_ in ["NOUN", "PROPN"]:
                                        head = self._get_span_with_compounds(token)
                                        tail = self._get_span_with_compounds(grandchild)
                                        
                                        # Clean head and tail
                                        head = re.sub(r'[^\w\s\-]', ' ', head).strip()
                                        tail = re.sub(r'[^\w\s\-]', ' ', tail).strip()
                                        
                                        if len(head) > 2 and len(tail) > 2:
                                            relationships.append({
                                                'source': head,
                                                'target': tail,
                                                'relationship': child.text,
                                                'confidence': 0.6  # Slightly lower confidence than S-V-O
                                            })
        print('relationships')
        return relationships
    
    def _get_span_with_compounds(self, token):
        """Get the full span including compound words"""
        start = token
        while any(child.dep_ == "compound" for child in start.children):
            for child in start.children:
                if child.dep_ == "compound" and child.i < start.i:
                    start = child
                    break
            else:
                break
                
        end = token
        while any(child.dep_ == "compound" for child in end.children):
            for child in end.children:
                if child.dep_ == "compound" and child.i > end.i:
                    end = child
                    break
            else:
                break
        
        return " ".join(t.text for t in token.doc[start.i:end.i+1])
    
    def create_hierarchical_topic_structure(self, topics):
        """Create a hierarchical structure of topics and subtopics with improved coherence"""
        # Calculate similarity between topics based on their terms and sentences
        topic_embeddings = []
        
        for topic in topics:
            # Create embedding from representative sentence and terms
            topic_text = topic['representative_sentence'] + " " + " ".join(topic['terms'][:5])
            embedding = self.sentence_model.encode([topic_text])[0]
            topic_embeddings.append(embedding)
            
        topic_embeddings = np.array(topic_embeddings)
        similarity_matrix = cosine_similarity(topic_embeddings)
        
        # Create hierarchical structure
        hierarchy = {}
        main_topics = []
        subtopics = set()
        
        # Sort topics by importance and semantic coherence
        sorted_topics = sorted(topics, key=lambda x: x['importance'], reverse=True)
        
        # First pass: identify main topics using importance and semantic coherence
        min_main_topics = max(10, len(topics) // 5)  # Ensure we have a reasonable number of main topics
        max_main_topics = min(10,11)  # Limit to avoid too many main topics
        
        # Add top N topics as main topics based on importance
        top_topics_by_importance = sorted_topics[:min_main_topics]
        for topic in top_topics_by_importance:
            main_topics.append(topic)
            print(topic)
        
        # For remaining topics, evaluate if they should be main topics or subtopics
        for i, topic in enumerate(sorted_topics[min_main_topics:]):
            # Skip if we've already reached our max main topics
            if len(main_topics) >= max_main_topics:
                subtopics.add(topic['id'])
                continue
                
            # Calculate semantic similarity to existing main topics
            max_similarity = 0
            for main_topic in main_topics:
                idx1 = next(idx for idx, t in enumerate(topics) if t['id'] == topic['id'])
                idx2 = next(idx for idx, t in enumerate(topics) if t['id'] == main_topic['id'])
                sim = similarity_matrix[idx1][idx2]
                max_similarity = max(max_similarity, sim)
            
            # If this topic is sufficiently distinct from existing main topics
            # and has high importance, make it a main topic
            if max_similarity < 0.4 and topic['importance'] > 5:
                main_topics.append(topic)
            else:
                subtopics.add(topic['id'])
        
        # Second pass: assign subtopics to main topics with improved coherence
        for main_topic in main_topics:
            hierarchy[main_topic['id']] = []
            main_idx = next(idx for idx, t in enumerate(topics) if t['id'] == main_topic['id'])
            
            # Calculate similarity between this main topic and all potential subtopics
            similarities = []
            for i, topic in enumerate(topics):
                if topic['id'] in subtopics:
                    sim = similarity_matrix[main_idx][i]
                    similarities.append((topic['id'], sim))
            
            # Sort by similarity (most similar first)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Assign subtopics to this main topic based on highest similarity
            for subtopic_id, sim in similarities:
                # Only assign if similarity is above threshold and not already assigned
                if sim > 0.35 and not any(subtopic_id in hierarchy[mt_id] for mt_id in hierarchy):
                    hierarchy[main_topic['id']].append(subtopic_id)
        
        # Ensure all subtopics are assigned to a parent
        # For any unassigned subtopics, find the best matching main topic
        unassigned = subtopics - set(subtopic_id for main_id in hierarchy for subtopic_id in hierarchy[main_id])
        for subtopic_id in unassigned:
            best_main_topic = None
            best_similarity = 0
            subtopic_idx = next(idx for idx, t in enumerate(topics) if t['id'] == subtopic_id)
            
            for main_topic in main_topics:
                main_idx = next(idx for idx, t in enumerate(topics) if t['id'] == main_topic['id'])
                sim = similarity_matrix[subtopic_idx][main_idx]
                if sim > best_similarity:
                    best_similarity = sim
                    best_main_topic = main_topic['id']
            
            if best_main_topic:
                hierarchy[best_main_topic].append(subtopic_id)
        
        # Third pass: analyze cross-topic relationships
        cross_topic_relationships = []
        
        # Create a relationship graph for transitive coherence
        relationship_graph = nx.Graph()
        for i, topic1 in enumerate(topics):
            relationship_graph.add_node(topic1['id'])
            for j, topic2 in enumerate(topics):
                if i != j and similarity_matrix[i][j] > 0.3:
                    relationship_graph.add_edge(topic1['id'], topic2['id'], weight=similarity_matrix[i][j])
        
        # Find community structure to enhance coherence
        communities = nx.community.greedy_modularity_communities(relationship_graph, weight='weight')
        
        # Use community structure to enhance relationships between topics in the same community
        for community in communities:
            community_nodes = list(community)
            for i, node1 in enumerate(community_nodes):
                for j, node2 in enumerate(community_nodes[i+1:], i+1):
                    if relationship_graph.has_edge(node1, node2):
                        # Boost similarity for nodes in the same community
                        sim = relationship_graph[node1][node2]['weight'] * 1.2  # 20% boost
                        
                        # Skip relationships between main topics and their subtopics (already handled)
                        if node1 in hierarchy and node2 in hierarchy[node1]:
                            continue
                        if node2 in hierarchy and node1 in hierarchy[node2]:
                            continue
                        
                        cross_topic_relationships.append({
                            'source': node1,
                            'target': node2,
                            'similarity': min(sim, 0.99)  # Cap at 0.99
                        })
        
        # Sort cross-topic relationships by similarity
        cross_topic_relationships = sorted(cross_topic_relationships, key=lambda x: x['similarity'], reverse=True)
        
        # Limit to the most significant relationships to avoid clutter
        max_cross_relationships = min(len(cross_topic_relationships), len(topics) * 2)
        cross_topic_relationships = cross_topic_relationships[:max_cross_relationships]
        
        return hierarchy, main_topics, [t for t in topics if t['id'] in subtopics], cross_topic_relationships


    def create_knowledge_graph(self, text):
        # Extract topics
        topics = self.extract_topics_and_subtopics(text)
        
        # Create hierarchical structure
        hierarchy, main_topics, subtopics, cross_topic_relationships = self.create_hierarchical_topic_structure(topics)
        
        # Initialize graph
        nodes = []
        edges = []
        
        # Add document root node
        root_id = 'doc_root'
        nodes.append({
            'id': root_id,
            'label': 'Document',
            'type': 'DOCUMENT',
            'importance': 10
        })
        
        # Process topics
        all_topic_ids = set()
        topic_map = {topic['id']: topic for topic in topics}
        
        # Track max importance for normalization
        max_importance = max(topic['importance'] for topic in topics) if topics else 10
        
        # Add main topics connected to root
        for topic in main_topics:
            topic_id = topic['id']
            all_topic_ids.add(topic_id)
            
            # Add node
            nodes.append({
                'id': topic_id,
                'label': topic['label'],
                'type': 'MAIN_TOPIC',
                'importance': topic['importance'],
                'terms': topic['terms'][:10]  # Store top 5 terms for reference
            })
            
            # Connect to root with normalized weight
            edges.append({
                'source': root_id,
                'target': topic_id,
                'relationship': 'CONTAINS',
                'weight': round((topic['importance'] / max_importance) * 10, 1)  # Scale to 0-10
            })
            
            # Add subtopics if present
            if topic_id in hierarchy and hierarchy[topic_id]:
                for subtopic_id in hierarchy[topic_id]:
                    if subtopic_id in topic_map and subtopic_id not in all_topic_ids:
                        subtopic = topic_map[subtopic_id]
                        all_topic_ids.add(subtopic_id)
                        
                        # Add subtopic node
                        nodes.append({
                            'id': subtopic_id,
                            'label': subtopic['label'],
                            'type': 'SUBTOPIC',
                            'importance': subtopic['importance'] * 0.8,  # Slightly smaller than main topics
                            'terms': subtopic['terms'][:10]  # Store top 5 terms (increased from 3)
                        })
                        
                        # Connect to main topic with normalized weight
                        edges.append({
                            'source': topic_id,
                            'target': subtopic_id,
                            'relationship': 'HAS_SUBTOPIC',
                            'weight': round((subtopic['importance'] / max_importance) * 8, 1)  # Scaled to 0-8
                        })
            
            # Extract entities, concepts and domain-specific terms for this topic
            topic_sentences = topic['sentences']
            entities, concepts, domain_terms = self.extract_entities_and_concepts(topic_sentences)
            
            # Add important entities with better categorization
            entity_count = 0
            entity_items = sorted(entities.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Calculate max entity score for normalization
            max_entity_score = max([e['score'] for _, e in entity_items[:5]]) if entity_items else 1
            
            for key, entity in entity_items:
                if entity_count >= 10:  # Limit to top 5 entities per topic
                    break
                    
                if entity['count'] >= 2:  # Must appear at least twice
                    entity_id = f"entity_{topic_id}_{entity_count}"
                    
                    # Categorize entity type more specifically
                    category = self._categorize_entity(entity['text'], entity['type'])
                    
                    # Add entity node
                    nodes.append({
                        'id': entity_id,
                        'label': entity['text'],
                        'type': 'ENTITY',
                        'entity_type': category,
                        'importance': min(8, entity['score']),
                        'mentions': entity['mentions'][:2]  # Store sample mentions for context
                    })
                    
                    # Connect to topic with normalized weight
                    edges.append({
                        'source': topic_id,
                        'target': entity_id,
                        'relationship': 'HAS_ENTITY',
                        'weight': round((entity['score'] / max_entity_score) * 7, 1)  # Scale to 0-7
                    })
                    
                    entity_count += 1
            
            # Add important concepts
            concept_count = 0
            concept_items = sorted(concepts.items(), key=lambda x: x[1]['score'], reverse=True)
            
            # Calculate max concept score for normalization
            max_concept_score = max([c['score'] for _, c in concept_items[:7]]) if concept_items else 1
            
            for key, concept in concept_items:
                if concept_count >= 7:  # Limit to top 7 concepts per topic
                    break
                    
                if concept['count'] >= 2:  # Must appear at least twice
                    concept_id = f"concept_{topic_id}_{concept_count}"
                    
                    # Categorize concept
                    concept_category = self._categorize_concept(concept['text'])
                    
                    # Add concept node
                    nodes.append({
                        'id': concept_id,
                        'label': concept['text'].title(),
                        'type': 'CONCEPT',
                        'concept_type': concept_category,
                        'importance': min(6, concept['score']),
                        'mentions': concept['mentions'][:2]  # Store sample mentions for context
                    })
                    
                    # Connect to topic with normalized weight
                    edges.append({
                        'source': topic_id,
                        'target': concept_id,
                        'relationship': 'HAS_CONCEPT',
                        'weight': round((concept['score'] / max_concept_score) * 6, 1)  # Scale to 0-6
                    })
                    
                    concept_count += 1
            
            # Add domain-specific terminology
            term_count = 0
        domain_term_items = sorted(domain_terms.items(), key=lambda x: x[1]['rank'], reverse=True)
        
        # Calculate max term rank for normalization
        max_term_rank = max([t['rank'] for _, t in domain_term_items[:7]]) if domain_term_items else 1
        
        # Dictionary to track terms we've already added to avoid near-duplicates
        added_terms = set()
        
        for key, term in domain_term_items:
            if term_count >= 5:  # Dynamic capacity based on topic importance
                break
                
            # Improved filtering for quality domain terms
            if term['count'] >= 1 and term['rank'] > 0.01:
                # Skip terms that are too similar to ones we've already added
                too_similar = False
                for added_term in added_terms:
                    # Check if current term is very similar to an existing one
                    if (key in added_term or added_term in key) and len(key) > 0.7 * len(added_term):
                        too_similar = True
                        break
                    
                if not too_similar:
                    term_id = f"term_{topic_id}_{term_count}"
                    
                    # Better label formatting - if multi-word term, ensure proper capitalization
                    label = ' '.join(w.capitalize() if not w.islower() else w for w in term['text'].split())
                    
                    # Add term node
                    nodes.append({
                        'id': term_id,
                        'label': label,
                        'type': 'DOMAIN_TERM',
                        'term_type': self._categorize_domain_term(term['text']),
                        'importance': min(5.5, term['rank'] * 12),  # Increased scaling for terms
                        'mentions': term['mentions'][:1] if term['mentions'] else []
                    })
                    
                    # Connect to topic with better weighting
                    edges.append({
                        'source': topic_id,
                        'target': term_id,
                        'relationship': 'CONTAINS_TERM',
                        'weight': round((term['rank'] / max_term_rank) * 5.5, 1)  # Increased max weight
                    })
                    
                    # Track this term to avoid similar ones
                    added_terms.add(key)
                    term_count += 1
            
        # Extract relationships between entities, concepts and terms
        all_sentences = [s for t in topics for s in t['sentences']]
        relationships = self.extract_relationships(all_sentences)
        
        # Add significant relationships between entities and concepts
        added_relationships = set()  # To avoid duplicates
        
        for rel in relationships:
            source_text = rel['source'].lower()
            target_text = rel['target'].lower()
            
            # Find nodes with matching labels
            source_nodes = [n for n in nodes if n['label'].lower() == source_text and n['type'] in ['ENTITY', 'CONCEPT', 'DOMAIN_TERM']]
            target_nodes = [n for n in nodes if n['label'].lower() == target_text and n['type'] in ['ENTITY', 'CONCEPT', 'DOMAIN_TERM']]
            
            for source_node in source_nodes:
                for target_node in target_nodes:
                    rel_key = f"{source_node['id']}_{target_node['id']}"
                    
                    if rel_key not in added_relationships:
                        edges.append({
                            'source': source_node['id'],
                            'target': target_node['id'],
                            'relationship': rel['relationship'].upper(),
                            'weight': round(rel['confidence'] * 5, 1)  # Scale confidence to weight
                        })
                        added_relationships.add(rel_key)
        
        # Add cross-topic relationships
        added_cross_topic = set()
        max_cross_edges = min(len(cross_topic_relationships), len(nodes) // 3)  # Limit to avoid visual clutter
        sorted_cross_relationships = sorted(cross_topic_relationships, key=lambda x: x['similarity'], reverse=True)

        for idx, rel in enumerate(sorted_cross_relationships):
            if idx >= max_cross_edges:
                break
                
            if rel['similarity'] > 0.4:  # Higher threshold for better coherence
                source_id = rel['source']
                target_id = rel['target']
                rel_key = f"{source_id}_{target_id}"
                rev_key = f"{target_id}_{source_id}"
                
                if rel_key not in added_relationships and rev_key not in added_relationships:
                    # Get source and target nodes to check their types
                    source_type = next((n['type'] for n in nodes if n['id'] == source_id), None)
                    target_type = next((n['type'] for n in nodes if n['id'] == target_id), None)
                    
                    # Define relationship based on node types for more meaningful relationships
                    if source_type == 'MAIN_TOPIC' and target_type == 'MAIN_TOPIC':
                        relationship = 'RELATED_TO'
                    elif source_type in ['MAIN_TOPIC', 'SUBTOPIC'] and target_type in ['MAIN_TOPIC', 'SUBTOPIC']:
                        # Find more descriptive relationship for topic-to-topic connections
                        source_topic = next((t for t in topics if t['id'] == source_id), None)
                        target_topic = next((t for t in topics if t['id'] == target_id), None)
                        
                        if source_topic and target_topic:
                            # Get common terms to determine relationship type
                            common_terms = set(source_topic['terms']).intersection(set(target_topic['terms']))
                            if common_terms:
                                relationship = 'SHARES_CONCEPTS'
                            else:
                                relationship = 'COMPLEMENTARY'
                        else:
                            relationship = 'RELATED_TO'
                    else:
                        relationship = 'RELATED_TO'
                    
                    edges.append({
                        'source': source_id,
                        'target': target_id,
                        'relationship': relationship,
                        'weight': round(rel['similarity'] * 8, 1),  # Scale to 0-8
                        'coherence': True  # Mark as coherence-enhancing edge
                    })
                    
                    added_relationships.add(rel_key)
                    added_cross_topic.add(rel_key)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'topic_count': len(main_topics),
                'subtopic_count': len(subtopics),
                'entity_count': len([n for n in nodes if n['type'] == 'ENTITY']),
                'concept_count': len([n for n in nodes if n['type'] == 'CONCEPT']),
                'term_count': len([n for n in nodes if n['type'] == 'DOMAIN_TERM']),
                'relationship_count': len(edges) - len(main_topics) - sum(len(v) for v in hierarchy.values()),
                'text_length': len(" ".join(all_sentences))
            }
        }
    def enhance_topic_coherence(self, topics):
        """Improve coherence between topics by refining terminology and relationships"""
        # 1. Create a unified vocabulary across topics
        all_terms = {}
        
        # Collect all terms from all topics
        for topic in topics:
            for term in topic['terms']:
                if term not in all_terms:
                    all_terms[term] = {
                        'count': 1,
                        'topics': [topic['id']]
                    }
                else:
                    all_terms[term]['count'] += 1
                    all_terms[term]['topics'].append(topic['id'])
        
        # 2. Normalize terminology across topics
        # Find synonyms and similar terms
        similar_terms = {}
        terms_list = list(all_terms.keys())
        
        for i, term1 in enumerate(terms_list):
            for term2 in terms_list[i+1:]:
                # Skip very short terms
                if len(term1) < 4 or len(term2) < 4:
                    continue
                    
                # Check if one term contains the other
                if term1 in term2 or term2 in term1:
                    # The longer term is likely more specific
                    canonical = term1 if len(term1) > len(term2) else term2
                    variant = term2 if canonical == term1 else term1
                    
                    if canonical not in similar_terms:
                        similar_terms[canonical] = [variant]
                    else:
                        similar_terms[canonical].append(variant)
        
        # 3. Update topics with normalized terminology
        for topic in topics:
            updated_terms = []
            term_map = {}  # Maps original terms to canonical terms
            
            for term in topic['terms']:
                # Check if this term is a variant of a canonical term
                is_variant = False
                for canonical, variants in similar_terms.items():
                    if term in variants:
                        term_map[term] = canonical
                        if canonical not in updated_terms:
                            updated_terms.append(canonical)
                        is_variant = True
                        break
                
                # If not a variant, keep the original term
                if not is_variant:
                    updated_terms.append(term)
            
            # Update topic terms with normalized terminology
            topic['terms'] = updated_terms[:10]  # Keep top 10 terms
            
            # Also replace terms in the topic label if needed
            for original, canonical in term_map.items():
                if original in topic['label']:
                    topic['label'] = topic['label'].replace(original, canonical)
        
        # 4. Ensure topic labels are distinct and informative
        topic_labels = {}
        for topic in topics:
            base_label = topic['label']
            
            # Check if this label is already used
            if base_label in topic_labels:
                # Find a differentiating term from the terms list
                for term in topic['terms']:
                    if term not in base_label:
                        # Add the term to make the label unique
                        topic['label'] = f"{base_label}: {term.title()}"
                        break
                
                # If still not unique, add a numeric identifier
                if topic['label'] in topic_labels:
                    topic['label'] = f"{base_label} ({len(topic_labels) + 1})"
            
            topic_labels[topic['label']] = topic['id']
        
        return topics
    def _categorize_domain_term(self, text):
        """Categorize domain-specific terminology into more meaningful categories"""
        text_lower = text.lower()
        
        # Check for method/algorithm patterns
        if re.search(r'(?i)(algorithm|method|technique|approach|procedure|protocol)', text_lower):
            return "METHOD"
        
        # Check for metrics/measurements
        elif re.search(r'(?i)(metric|measure|measurement|score|rate|ratio|index|coefficient)', text_lower):
            return "METRIC"
        
        # Check for models/frameworks
        elif re.search(r'(?i)(model|framework|architecture|system|structure|pattern)', text_lower):
            return "MODEL"
        
        # Check for concepts/theories
        elif re.search(r'(?i)(theory|concept|principle|paradigm|law|rule)', text_lower):
            return "THEORY"
        
        # Check for properties/attributes
        elif re.search(r'(?i)(property|attribute|characteristic|feature|aspect|quality)', text_lower):
            return "PROPERTY"
        
        # Check for processes/phenomena
        elif re.search(r'(?i)(process|effect|phenomenon|behavior|mechanism)', text_lower):
            return "PROCESS"
        
        # Check for tools/technologies
        elif re.search(r'(?i)(tool|technology|software|platform|device|application)', text_lower):
            return "TECHNOLOGY"
        
        # Analyze POS patterns as fallback
        doc = self.nlp(text)
        
        # Check if it contains specific POS patterns
        if any(token.pos_ == "VERB" for token in doc):
            return "ACTION"
        elif any(token.pos_ == "ADJ" for token in doc):
            return "DESCRIPTOR"
        else:
            return "DOMAIN_CONCEPT"
    def _categorize_entity(self, text, spacy_type):
        """Categorize entity into more specific types"""
        if spacy_type == "PERSON":
            return "PERSON"
        elif spacy_type == "ORG":
            return "ORGANIZATION"
        elif spacy_type == "GPE" or spacy_type == "LOC":
            return "LOCATION"
        elif spacy_type == "DATE" or spacy_type == "TIME":
            return "TEMPORAL"
        elif spacy_type == "PRODUCT":
            return "PRODUCT"
        elif spacy_type == "EVENT":
            return "EVENT"
        elif spacy_type == "WORK_OF_ART":
            return "CREATIVE_WORK"
        elif spacy_type == "LAW":
            return "LEGAL"
        elif spacy_type == "MONEY" or spacy_type == "PERCENT" or spacy_type == "QUANTITY":
            return "MEASURE"
        else:
            # Try to infer type from text pattern
            if re.search(r'\b\d{4}\b', text):  # Years
                return "TEMPORAL"
            elif re.search(r'\$|\€|\£|\¥', text):  # Currency
                return "FINANCIAL"
            elif re.search(r'(?i)(algorithm|function|method|technique)', text):
                return "METHOD"
            elif re.search(r'(?i)(theory|principle|law|rule)', text):
                return "THEORY"
            else:
                return "GENERAL"
    
    def _categorize_concept(self, text):
        """Categorize concept into semantic categories"""
        # Use simple keyword matching for basic categorization
        text_lower = text.lower()
        
        if re.search(r'(?i)(process|procedure|methodology|steps|workflow)', text_lower):
            return "PROCESS"
        elif re.search(r'(?i)(system|framework|structure|architecture)', text_lower):
            return "SYSTEM"
        elif re.search(r'(?i)(theory|concept|principle|idea|paradigm)', text_lower):
            return "THEORY"
        elif re.search(r'(?i)(tool|instrument|device|equipment)', text_lower):
            return "TOOL"
        elif re.search(r'(?i)(property|characteristic|attribute|feature)', text_lower):
            return "PROPERTY"
        elif re.search(r'(?i)(result|outcome|effect|impact)', text_lower):
            return "OUTCOME"
        elif re.search(r'(?i)(problem|challenge|issue|difficulty)', text_lower):
            return "PROBLEM"
        elif re.search(r'(?i)(solution|approach|strategy)', text_lower):
            return "SOLUTION"
        else:
            # Fallback - analyze POS pattern
            doc = self.nlp(text)
            if any(token.pos_ == "VERB" for token in doc):
                return "ACTION"
            elif any(token.pos_ == "ADJ" for token in doc):
                return "PROPERTY"
            else:
                return "OBJECT"

def generate_enhanced_knowledge_graph(file_path):
    """Generate enhanced knowledge graph from PDF content."""
    try:
        text = extract_text_from_pdf(file_path)
        extractor = EnhancedKnowledgeGraphExtractor()
        graph = extractor.create_knowledge_graph(text)
        return graph
    except Exception as e:
        print(f"Error generating knowledge graph: {str(e)}")
        return None
# from spacy import load
# import networkx as nx
# from textblob import TextBlob
# from app.services.pdf_processing import extract_text_from_pdf
# import spacy
# from collections import defaultdict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np
# import re
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# class EnhancedKnowledgeGraphExtractor:
#     def __init__(self):
#         self.nlp = spacy.load("en_core_web_sm")
#         self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
#         print('init')
        
#     def preprocess_text(self, text):
#         """Clean and preprocess the text"""
#         # Remove excessive whitespace
#         text = re.sub(r'\s+', ' ', text)
        
#         # Remove references like [1], [2], etc.
#         text = re.sub(r'\[\d+\]', '', text)
        
#         # Clean more problematic characters that could affect parsing
#         text = re.sub(r'[\u2022\u2023\u2043\u204C\u204D\u2219\u25D8\u25E6\u2619\u2765\u2767]', ' ', text)  # Bullets and special markers
        
#         # Split into proper sentences
#         doc = self.nlp(text)
#         sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
        
#         # Additional cleaning for each sentence
#         cleaned_sentences = []
#         for sentence in sentences:
#             # Remove broken token sequences like "F •(E"
#             clean_sent = re.sub(r'[A-Z]\s*[•\(\.]\s*\([A-Z]', '', sentence)
#             clean_sent = re.sub(r'[^\w\s\.,:;\-\(\)\/]', ' ', clean_sent)  # Keep some useful punctuation
#             clean_sent = re.sub(r'\s+', ' ', clean_sent).strip()
#             if len(clean_sent) > 20:  # Keep only meaningful sentences
#                 cleaned_sentences.append(clean_sent)
                
#         print('preprocess')
#         return cleaned_sentences
        
#     def extract_topics_and_subtopics(self, text):
#         # Preprocess and split text into sentences
#         sentences = self.preprocess_text(text)
        
#         if len(sentences) < 10:
#             return [{
#                 'id': 'topic_0',
#                 'label': 'Main Content',
#                 'type': 'MAIN_TOPIC',
#                 'sentences': sentences,
#                 'terms': [],
#                 'importance': 10
#             }]
        
#         # Calculate number of topics based on text length
#         n_clusters = min(max(3, len(sentences) // 15), 10)  # Adjusted ratio and max
        
#         # Create TF-IDF vectorizer
#         vectorizer = TfidfVectorizer(
#             max_features=300, 
#             stop_words='english',
#             ngram_range=(1, 3),  # Include trigrams for better phrases
#             min_df=2  # At least appear twice
#         )
#         tfidf_matrix = vectorizer.fit_transform(sentences)
        
#         # Get sentence embeddings for semantic similarity
#         embeddings = self.sentence_model.encode(sentences)
        
#         # Combine TF-IDF and semantic embeddings for better clustering
#         # Use KMeans instead of AgglomerativeClustering to avoid issues
#         combined_features = np.hstack((
#             0.4 * tfidf_matrix.toarray(),  # TF-IDF features (40% weight)
#             0.6 * embeddings  # Semantic features (60% weight)
#         ))
        
#         # Use KMeans which is more robust across sklearn versions
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         clusters = kmeans.fit_predict(combined_features)
        
#         # Get important terms for each cluster
#         terms = vectorizer.get_feature_names_out()
        
#         # For each cluster, find the most representative sentences and terms
#         topics = []
#         for i in range(n_clusters):
#             cluster_sentences = [sent for j, sent in enumerate(sentences) if clusters[j] == i]
            
#             if not cluster_sentences:
#                 continue
                
#             # Get documents in this cluster
#             cluster_indices = [j for j, c in enumerate(clusters) if c == i]
            
#             # Calculate average TF-IDF vector for this cluster
#             cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
#             cluster_tfidf = np.asarray(cluster_tfidf).flatten()
            
#             # Get top terms and clean them
#             top_term_indices = cluster_tfidf.argsort()[-15:][::-1]  # Get more terms initially for filtering
#             raw_topic_terms = [terms[idx] for idx in top_term_indices]
            
#             # Clean and filter terms to remove problematic terms
#             topic_terms = []
#             for term in raw_topic_terms:
#                 # Skip terms with too many special characters or very short terms
#                 clean_term = re.sub(r'[^\w\s\-]', '', term).strip()
#                 if len(clean_term) < 3 or not re.search(r'[a-zA-Z]', clean_term):
#                     continue
#                 # Avoid duplicate terms
#                 if clean_term and clean_term not in topic_terms:
#                     topic_terms.append(clean_term)
            
#             # Find most central sentence as the representative
#             central_sentence_idx = None
#             max_similarity = -1
            
#             # Calculate pairwise similarities within the cluster
#             cluster_embeddings = embeddings[cluster_indices]
#             similarities = cosine_similarity(cluster_embeddings)
            
#             # Find the sentence with highest average similarity to others
#             for idx, sent_idx in enumerate(cluster_indices):
#                 avg_similarity = similarities[idx].mean()
#                 if avg_similarity > max_similarity:
#                     max_similarity = avg_similarity
#                     central_sentence_idx = sent_idx
            
#             # Use most central sentence and top terms to create a meaningful label
#             representative_sentence = sentences[central_sentence_idx] if central_sentence_idx is not None else ""
            
#             # Extract noun phrases from central sentence
#             central_doc = self.nlp(representative_sentence)
#             noun_phrases = [chunk.text for chunk in central_doc.noun_chunks if len(chunk.text) > 3]
            
#             # Create label from noun phrases or top terms
#             if noun_phrases:
#                 # Clean the noun phrase
#                 label = re.sub(r'[^\w\s\-]', ' ', noun_phrases[0])
#                 label = re.sub(r'\s+', ' ', label).strip().title()
#             else:
#                 # Filter terms to get better phrases (prefer multi-word terms)
#                 multi_word_terms = [term for term in topic_terms if ' ' in term]
#                 if multi_word_terms:
#                     label = multi_word_terms[0].title()
#                 else:
#                     label = ' '.join(topic_terms[:2]).title() if topic_terms else f"Topic {i}"
            
#             # Ensure label is meaningful - minimum length and sensible content
#             if len(label) < 3 or not re.search(r'[a-zA-Z]', label):
#                 label = f"Topic {i+1}: {topic_terms[0].title() if topic_terms else ''}"
            
#             # Calculate importance based on cluster size and semantic coherence
#             cluster_size = len(cluster_sentences)
#             coherence = max_similarity  # Use the max similarity as coherence measure
            
#             # Combined importance score
#             importance = (cluster_size / len(sentences) * 7) + (coherence * 3)
            
#             topics.append({
#                 'id': f'topic_{i}',
#                 'label': label,
#                 'type': 'MAIN_TOPIC',
#                 'sentences': cluster_sentences,
#                 'terms': topic_terms[:10],  # Keep top 10 terms after cleaning
#                 'importance': importance,
#                 'representative_sentence': representative_sentence
#             })
#         print('topics')
#         return topics
    
#     def extract_entities_and_concepts(self, sentences):
#         """Extract entities and concepts from text with improved processing"""
#         combined_text = " ".join(sentences)
#         doc = self.nlp(combined_text)
        
#         entities = {}
#         concepts = {}
        
#         # Process named entities
#         for ent in doc.ents:
#             text = ent.text.strip()
#             if len(text) > 2:
#                 # Clean the entity text
#                 text = re.sub(r'[^\w\s\-]', ' ', text)
#                 text = re.sub(r'\s+', ' ', text).strip()
                
#                 if not text or len(text) < 3 or not re.search(r'[a-zA-Z]', text):
#                     continue
                
#                 key = text.lower()
                
#                 if key not in entities:
#                     entities[key] = {
#                         'text': text,
#                         'type': ent.label_,
#                         'count': 0,
#                         'mentions': []
#                     }
                    
#                 entities[key]['count'] += 1
#                 context = doc[max(0, ent.start - 5):min(len(doc), ent.end + 5)].text
#                 # Clean the context
#                 context = re.sub(r'[^\w\s\-\.,;:\(\)]', ' ', context)
#                 context = re.sub(r'\s+', ' ', context).strip()
#                 entities[key]['mentions'].append(context)
        
#         # Process noun phrases for concepts
#         for chunk in doc.noun_chunks:
#             text = chunk.text.strip()
#             if len(text) > 3:
#                 # Clean the concept text
#                 text = re.sub(r'[^\w\s\-]', ' ', text)
#                 text = re.sub(r'\s+', ' ', text).strip()
                
#                 if not text or len(text) < 3 or not re.search(r'[a-zA-Z]', text):
#                     continue
                
#                 key = text.lower()
                
#                 # Check if this is not already tracked as an entity
#                 if key not in entities and key not in concepts:
#                     concepts[key] = {
#                         'text': text,
#                         'count': 0,
#                         'mentions': []
#                     }
                
#                 if key in concepts:
#                     concepts[key]['count'] += 1
#                     context = doc[max(0, chunk.start - 5):min(len(doc), chunk.end + 5)].text
#                     # Clean the context
#                     context = re.sub(r'[^\w\s\-\.,;:\(\)]', ' ', context)
#                     context = re.sub(r'\s+', ' ', context).strip()
#                     concepts[key]['mentions'].append(context)
        
#         print('extract entities')
#         return entities, concepts

#     def extract_relationships(self, sentences):
#         """Extract semantic relationships between entities"""
#         relationships = []
        
#         for sentence in sentences:
#             # Clean the sentence to avoid parsing issues
#             clean_sentence = re.sub(r'[^\w\s\-\.,;:\(\)]', ' ', sentence)
#             clean_sentence = re.sub(r'\s+', ' ', clean_sentence).strip()
            
#             doc = self.nlp(clean_sentence)
            
#             # Extract subject-verb-object patterns
#             for sent in doc.sents:
#                 entities = [(e.text, e.start, e.label_) for e in sent.ents]
#                 if len(entities) < 2:
#                     continue
                    
#                 # Find verb-mediated relationships
#                 for token in sent:
#                     if token.dep_ == "ROOT" and token.pos_ == "VERB":
#                         subject = None
#                         direct_object = None
                        
#                         # Find subject
#                         for child in token.children:
#                             if child.dep_ in ["nsubj", "nsubjpass"]:
#                                 subject = self._get_span_with_compounds(child)
#                                 break
                        
#                         # Find object
#                         for child in token.children:
#                             if child.dep_ in ["dobj", "pobj"]:
#                                 direct_object = self._get_span_with_compounds(child)
#                                 break
                        
#                         if subject and direct_object:
#                             # Clean subject and object
#                             subject = re.sub(r'[^\w\s\-]', ' ', subject).strip()
#                             direct_object = re.sub(r'[^\w\s\-]', ' ', direct_object).strip()
                            
#                             if len(subject) > 2 and len(direct_object) > 2:
#                                 relationships.append({
#                                     'source': subject,
#                                     'target': direct_object,
#                                     'relationship': token.lemma_
#                                 })
#         print('rel')
#         return relationships
    
#     def _get_span_with_compounds(self, token):
#         """Get the full span including compound words"""
#         start = token
#         while any(child.dep_ == "compound" for child in start.children):
#             for child in start.children:
#                 if child.dep_ == "compound" and child.i < start.i:
#                     start = child
#                     break
#             else:
#                 break
                
#         end = token
#         while any(child.dep_ == "compound" for child in end.children):
#             for child in end.children:
#                 if child.dep_ == "compound" and child.i > end.i:
#                     end = child
#                     break
#             else:
#                 break
        
#         return " ".join(t.text for t in token.doc[start.i:end.i+1])
    
#     def create_hierarchical_topic_structure(self, topics):
#         """Create a hierarchical structure of topics and subtopics"""
#         # Calculate similarity between topics based on their terms and sentences
#         topic_embeddings = []
        
#         for topic in topics:
#             # Create embedding from representative sentence and terms
#             topic_text = topic['representative_sentence'] + " " + " ".join(topic['terms'][:5])
#             embedding = self.sentence_model.encode([topic_text])[0]
#             topic_embeddings.append(embedding)
            
#         topic_embeddings = np.array(topic_embeddings)
#         similarity_matrix = cosine_similarity(topic_embeddings)
        
#         # Create hierarchical structure
#         hierarchy = {}
#         main_topics = []
#         subtopics = set()
        
#         # Sort topics by importance
#         sorted_topics = sorted(topics, key=lambda x: x['importance'], reverse=True)
        
#         # First pass: identify main topics and potential subtopics
#         for i, topic in enumerate(sorted_topics):
#             is_subtopic = False
            
#             for j, potential_parent in enumerate(sorted_topics):
#                 if i != j and similarity_matrix[i][j] > 0.5 and topic['importance'] < potential_parent['importance']:
#                     # This topic is similar to a more important topic - mark as potential subtopic
#                     is_subtopic = True
#                     break
            
#             if not is_subtopic:
#                 main_topics.append(topic)
#             else:
#                 subtopics.add(topic['id'])
        
#         # Second pass: assign subtopics to main topics
#         for main_topic in main_topics:
#             hierarchy[main_topic['id']] = []
            
#             for i, topic in enumerate(sorted_topics):
#                 if topic['id'] in subtopics:  # Only consider topics marked as subtopics
#                     j = next((idx for idx, t in enumerate(sorted_topics) if t['id'] == main_topic['id']), None)
                    
#                     if j is not None and similarity_matrix[i][j] > 0.35:  # Threshold for parent-child similarity
#                         hierarchy[main_topic['id']].append(topic['id'])
        
#         # Third pass: analyze cross-topic relationships
#         cross_topic_relationships = []
#         for i, topic1 in enumerate(topics):
#             for j, topic2 in enumerate(topics):
#                 if i != j and similarity_matrix[i][j] > 0.3:  # Lower threshold to catch more relationships
#                     # Skip relationships between main topics and their subtopics (already handled)
#                     if topic1['id'] in main_topics and topic2['id'] in subtopics and topic2['id'] in hierarchy.get(topic1['id'], []):
#                         continue
#                     if topic2['id'] in main_topics and topic1['id'] in subtopics and topic1['id'] in hierarchy.get(topic2['id'], []):
#                         continue
                        
#                     cross_topic_relationships.append({
#                         'source': topic1['id'],
#                         'target': topic2['id'],
#                         'similarity': similarity_matrix[i][j]
#                     })
                        
#         print('heir')
#         return hierarchy, main_topics, [t for t in topics if t['id'] in subtopics], cross_topic_relationships

#     def create_knowledge_graph(self, text):
#         # Extract topics
#         topics = self.extract_topics_and_subtopics(text)
        
#         # Create hierarchical structure
#         hierarchy, main_topics, subtopics, cross_topic_relationships = self.create_hierarchical_topic_structure(topics)
        
#         # Initialize graph
#         nodes = []
#         edges = []
        
#         # Add document root node
#         root_id = 'doc_root'
#         nodes.append({
#             'id': root_id,
#             'label': 'Document',
#             'type': 'DOCUMENT',
#             'importance': 10
#         })
        
#         # Process topics
#         all_topic_ids = set()
#         topic_map = {topic['id']: topic for topic in topics}
        
#         # Track max importance for normalization
#         max_importance = max(topic['importance'] for topic in topics) if topics else 10
        
#         # Add main topics connected to root
#         for topic in main_topics:
#             topic_id = topic['id']
#             all_topic_ids.add(topic_id)
            
#             # Add node
#             nodes.append({
#                 'id': topic_id,
#                 'label': topic['label'],
#                 'type': 'MAIN_TOPIC',
#                 'importance': topic['importance'],
#                 'terms': topic['terms'][:5]  # Store top 5 terms for reference
#             })
            
#             # Connect to root with normalized weight
#             edges.append({
#                 'source': root_id,
#                 'target': topic_id,
#                 'relationship': 'CONTAINS',
#                 'weight': round((topic['importance'] / max_importance) * 10, 1)  # Scale to 0-10
#             })
            
#             # Add subtopics if present
#             if topic_id in hierarchy and hierarchy[topic_id]:
#                 for subtopic_id in hierarchy[topic_id]:
#                     if subtopic_id in topic_map and subtopic_id not in all_topic_ids:
#                         subtopic = topic_map[subtopic_id]
#                         all_topic_ids.add(subtopic_id)
                        
#                         # Add subtopic node
#                         nodes.append({
#                             'id': subtopic_id,
#                             'label': subtopic['label'],
#                             'type': 'SUBTOPIC',
#                             'importance': subtopic['importance'] * 0.8,  # Slightly smaller than main topics
#                             'terms': subtopic['terms'][:5]  # Store top 5 terms (increased from 3)
#                         })
                        
#                         # Connect to main topic with normalized weight
#                         edges.append({
#                             'source': topic_id,
#                             'target': subtopic_id,
#                             'relationship': 'HAS_SUBTOPIC',
#                             'weight': round((subtopic['importance'] / max_importance) * 8, 1)  # Scaled to 0-8
#                         })
            
#             # Extract entities and concepts for this topic
#             topic_sentences = topic['sentences']
#             entities, concepts = self.extract_entities_and_concepts(topic_sentences)
            
#             # Add important entities
#             entity_count = 0
#             entity_items = sorted(entities.items(), key=lambda x: x[1]['count'], reverse=True)
            
#             # Calculate max entity count for normalization
#             max_entity_count = max([e['count'] for _, e in entity_items[:5]]) if entity_items else 1
            
#             for key, entity in entity_items:
#                 if entity_count >= 5:  # Limit to top 5 entities per topic
#                     break
                    
#                 if entity['count'] >= 2:  # Must appear at least twice
#                     entity_id = f"entity_{topic_id}_{entity_count}"
                    
#                     # Add entity node
#                     nodes.append({
#                         'id': entity_id,
#                         'label': entity['text'],
#                         'type': 'ENTITY',
#                         'entity_type': entity['type'],
#                         'importance': min(8, entity['count'] * 1.5),
#                         'mentions': entity['mentions'][:2]  # Store sample mentions for context
#                     })
                    
#                     # Connect to topic with normalized weight
#                     edges.append({
#                         'source': topic_id,
#                         'target': entity_id,
#                         'relationship': 'HAS_ENTITY',
#                         'weight': round((entity['count'] / max_entity_count) * 7, 1)  # Scale to 0-7
#                     })
                    
#                     entity_count += 1
            
#             # Add important concepts
#             concept_count = 0
#             concept_items = sorted(concepts.items(), key=lambda x: x[1]['count'], reverse=True)
            
#             # Calculate max concept count for normalization
#             max_concept_count = max([c['count'] for _, c in concept_items[:7]]) if concept_items else 1
            
#             for key, concept in concept_items:
#                 if concept_count >= 7:  # Limit to top 7 concepts per topic
#                     break
                    
#                 if concept['count'] >= 2:  # Must appear at least twice
#                     concept_id = f"concept_{topic_id}_{concept_count}"
                    
#                     # Add concept node
#                     nodes.append({
#                         'id': concept_id,
#                         'label': concept['text'].title(),
#                         'type': 'CONCEPT',
#                         'importance': min(6, concept['count']),
#                         'mentions': concept['mentions'][:2]  # Store sample mentions for context
#                     })
                    
#                     # Connect to topic with normalized weight
#                     edges.append({
#                         'source': topic_id,
#                         'target': concept_id,
#                         'relationship': 'HAS_CONCEPT',
#                         'weight': round((concept['count'] / max_concept_count) * 6, 1)  # Scale to 0-6
#                     })
                    
#                     concept_count += 1
        
#         # Extract cross-topic relationships
#         relationships = self.extract_relationships([s for t in topics for s in t['sentences']])
        
#         # Add significant relationships between entities and concepts
#         added_relationships = set()  # To avoid duplicates
        
#         for rel in relationships:
#             source_text = rel['source'].lower()
#             target_text = rel['target'].lower()
            
#             # Find nodes with matching labels
#             source_nodes = [n for n in nodes if n['label'].lower() == source_text and n['type'] in ['ENTITY', 'CONCEPT']]
#             target_nodes = [n for n in nodes if n['label'].lower() == target_text and n['type'] in ['ENTITY', 'CONCEPT']]
            
#             for source_node in source_nodes:
#                 for target_node in target_nodes:
#                     rel_key = f"{source_node['id']}_{target_node['id']}"
                    
#                     if rel_key not in added_relationships:
#                         edges.append({
#                             'source': source_node['id'],
#                             'target': target_node['id'],
#                             'relationship': rel['relationship'].upper(),
#                             'weight': 3
#                         })
#                         added_relationships.add(rel_key)
        
#         # Add cross-topic relationships
#         for rel in cross_topic_relationships:
#             if rel['similarity'] > 0.4:  # Threshold for actual edge creation
#                 source_id = rel['source']
#                 target_id = rel['target']
#                 rel_key = f"{source_id}_{target_id}"
                
#                 if rel_key not in added_relationships:
#                     edges.append({
#                         'source': source_id,
#                         'target': target_id,
#                         'relationship': 'RELATED_TO',
#                         'weight': round(rel['similarity'] * 8, 1)  # Scale to 0-8
#                     })
#                     added_relationships.add(rel_key)
        
#         return {
#             'nodes': nodes,
#             'edges': edges,
#             'metadata': {
#                 'topic_count': len(main_topics),
#                 'subtopic_count': len(subtopics),
#                 'entity_count': len([n for n in nodes if n['type'] == 'ENTITY']),
#                 'concept_count': len([n for n in nodes if n['type'] == 'CONCEPT']),
#                 'relationship_count': len(edges)
#             }
#         }

# def generate_enhanced_knowledge_graph(file_path):
#     """Generate enhanced knowledge graph from PDF content."""
#     try:
#         text = extract_text_from_pdf(file_path)
#         extractor = EnhancedKnowledgeGraphExtractor()
#         graph = extractor.create_knowledge_graph(text)
#         return graph
#     except Exception as e:
#         print(f"Error generating knowledge graph: {str(e)}")
#         return None
# from spacy import load
# import networkx as nx
# from textblob import TextBlob
# from app.services.pdf_processing import extract_text_from_pdf
# import spacy
# from collections import defaultdict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np
# import re
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# class EnhancedKnowledgeGraphExtractor:
#     def __init__(self):
#         self.nlp = spacy.load("en_core_web_sm")
#         self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
#         print('init')
        
#     def preprocess_text(self, text):
#         """Clean and preprocess the text"""
#         # Remove excessive whitespace
#         text = re.sub(r'\s+', ' ', text)
        
#         # Remove references like [1], [2], etc.
#         text = re.sub(r'\[\d+\]', '', text)
        
#         # Split into proper sentences
#         doc = self.nlp(text)
#         sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 20]
#         print('preprocess')
#         return sentences
        
#     def extract_topics_and_subtopics(self, text):
#         # Preprocess and split text into sentences
#         sentences = self.preprocess_text(text)
        
#         if len(sentences) < 10:
#             return [{
#                 'id': 'topic_0',
#                 'label': 'Main Content',
#                 'type': 'MAIN_TOPIC',
#                 'sentences': sentences,
#                 'terms': [],
#                 'importance': 10
#             }]
        
#         # Calculate number of topics based on text length
#         n_clusters = min(max(3, len(sentences) // 15), 10)  # Adjusted ratio and max
        
#         # Create TF-IDF vectorizer
#         vectorizer = TfidfVectorizer(
#             max_features=300, 
#             stop_words='english',
#             ngram_range=(1, 3),  # Include trigrams for better phrases
#             min_df=2  # At least appear twice
#         )
#         tfidf_matrix = vectorizer.fit_transform(sentences)
        
#         # Get sentence embeddings for semantic similarity
#         embeddings = self.sentence_model.encode(sentences)
        
#         # Combine TF-IDF and semantic embeddings for better clustering
#         # Use KMeans instead of AgglomerativeClustering to avoid issues
#         combined_features = np.hstack((
#             0.4 * tfidf_matrix.toarray(),  # TF-IDF features (40% weight)
#             0.6 * embeddings  # Semantic features (60% weight)
#         ))
        
#         # Use KMeans which is more robust across sklearn versions
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         clusters = kmeans.fit_predict(combined_features)
        
#         # Get important terms for each cluster
#         terms = vectorizer.get_feature_names_out()
        
#         # For each cluster, find the most representative sentences and terms
#         topics = []
#         for i in range(n_clusters):
#             cluster_sentences = [sent for j, sent in enumerate(sentences) if clusters[j] == i]
            
#             if not cluster_sentences:
#                 continue
                
#             # Get documents in this cluster
#             cluster_indices = [j for j, c in enumerate(clusters) if c == i]
            
#             # Calculate average TF-IDF vector for this cluster
#             cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
#             cluster_tfidf = np.asarray(cluster_tfidf).flatten()
            
#             # Get top terms
#             top_term_indices = cluster_tfidf.argsort()[-10:][::-1]
#             topic_terms = [terms[idx] for idx in top_term_indices]
            
#             # Find most central sentence as the representative
#             central_sentence_idx = None
#             max_similarity = -1
            
#             # Calculate pairwise similarities within the cluster
#             cluster_embeddings = embeddings[cluster_indices]
#             similarities = cosine_similarity(cluster_embeddings)
            
#             # Find the sentence with highest average similarity to others
#             for idx, sent_idx in enumerate(cluster_indices):
#                 avg_similarity = similarities[idx].mean()
#                 if avg_similarity > max_similarity:
#                     max_similarity = avg_similarity
#                     central_sentence_idx = sent_idx
            
#             # Use most central sentence and top terms to create a meaningful label
#             representative_sentence = sentences[central_sentence_idx] if central_sentence_idx is not None else ""
            
#             # Extract noun phrases from central sentence
#             central_doc = self.nlp(representative_sentence)
#             noun_phrases = [chunk.text for chunk in central_doc.noun_chunks if len(chunk.text) > 3]
            
#             # Create label from noun phrases or top terms
#             if noun_phrases:
#                 label = noun_phrases[0]
#             else:
#                 # Filter terms to get better phrases (prefer multi-word terms)
#                 multi_word_terms = [term for term in topic_terms if ' ' in term]
#                 if multi_word_terms:
#                     label = multi_word_terms[0].title()
#                 else:
#                     label = ' '.join(topic_terms[:2]).title()
            
#             # Calculate importance based on cluster size and semantic coherence
#             cluster_size = len(cluster_sentences)
#             coherence = max_similarity  # Use the max similarity as coherence measure
            
#             # Combined importance score
#             importance = (cluster_size / len(sentences) * 7) + (coherence * 3)
            
#             topics.append({
#                 'id': f'topic_{i}',
#                 'label': label,
#                 'type': 'MAIN_TOPIC',
#                 'sentences': cluster_sentences,
#                 'terms': topic_terms,
#                 'importance': importance,
#                 'representative_sentence': representative_sentence
#             })
#         print('topics')
#         return topics
    
#     def extract_entities_and_concepts(self, sentences):
#         """Extract entities and concepts from text with improved processing"""
#         combined_text = " ".join(sentences)
#         doc = self.nlp(combined_text)
        
#         entities = {}
#         concepts = {}
        
#         # Process named entities
#         for ent in doc.ents:
#             if len(ent.text.strip()) > 2:
#                 text = ent.text.strip()
#                 key = text.lower()
                
#                 if key not in entities:
#                     entities[key] = {
#                         'text': text,
#                         'type': ent.label_,
#                         'count': 0,
#                         'mentions': []
#                     }
                    
#                 entities[key]['count'] += 1
#                 context = doc[max(0, ent.start - 5):min(len(doc), ent.end + 5)].text
#                 entities[key]['mentions'].append(context)
        
#         # Process noun phrases for concepts
#         for chunk in doc.noun_chunks:
#             if len(chunk.text.strip()) > 3 and not any(chunk.text.lower() == e.lower() for e in entities):
#                 text = chunk.text.strip()
#                 key = text.lower()
                
#                 if key not in concepts:
#                     concepts[key] = {
#                         'text': text,
#                         'count': 0,
#                         'mentions': []
#                     }
                
#                 concepts[key]['count'] += 1
#                 context = doc[max(0, chunk.start - 5):min(len(doc), chunk.end + 5)].text
#                 concepts[key]['mentions'].append(context)
#         print('extract entitites')
#         return entities, concepts

#     def extract_relationships(self, sentences):
#         """Extract semantic relationships between entities"""
#         relationships = []
        
#         for sentence in sentences:
#             doc = self.nlp(sentence)
            
#             # Extract subject-verb-object patterns
#             for sent in doc.sents:
#                 entities = [(e.text, e.start, e.label_) for e in sent.ents]
#                 if len(entities) < 2:
#                     continue
                    
#                 # Find verb-mediated relationships
#                 for token in sent:
#                     if token.dep_ == "ROOT" and token.pos_ == "VERB":
#                         subject = None
#                         direct_object = None
                        
#                         # Find subject
#                         for child in token.children:
#                             if child.dep_ in ["nsubj", "nsubjpass"]:
#                                 subject = self._get_span_with_compounds(child)
#                                 break
                        
#                         # Find object
#                         for child in token.children:
#                             if child.dep_ in ["dobj", "pobj"]:
#                                 direct_object = self._get_span_with_compounds(child)
#                                 break
                        
#                         if subject and direct_object:
#                             relationships.append({
#                                 'source': subject,
#                                 'target': direct_object,
#                                 'relationship': token.lemma_
#                             })
#         print('rel')
#         return relationships
    
#     def _get_span_with_compounds(self, token):
#         """Get the full span including compound words"""
#         start = token
#         while any(child.dep_ == "compound" for child in start.children):
#             for child in start.children:
#                 if child.dep_ == "compound" and child.i < start.i:
#                     start = child
#                     break
#             else:
#                 break
                
#         end = token
#         while any(child.dep_ == "compound" for child in end.children):
#             for child in end.children:
#                 if child.dep_ == "compound" and child.i > end.i:
#                     end = child
#                     break
#             else:
#                 break
#         print('span') 
#         return " ".join(t.text for t in token.doc[start.i:end.i+1])
    
#     def create_hierarchical_topic_structure(self, topics):
#         """Create a hierarchical structure of topics and subtopics"""
#         # Calculate similarity between topics based on their terms and sentences
#         topic_embeddings = []
        
#         for topic in topics:
#             # Create embedding from representative sentence and terms
#             topic_text = topic['representative_sentence'] + " " + " ".join(topic['terms'][:5])
#             embedding = self.sentence_model.encode([topic_text])[0]
#             topic_embeddings.append(embedding)
            
#         topic_embeddings = np.array(topic_embeddings)
#         similarity_matrix = cosine_similarity(topic_embeddings)
        
#         # Create hierarchical structure
#         hierarchy = {}
#         main_topics = []
#         subtopics = set()
        
#         # Sort topics by importance
#         sorted_topics = sorted(topics, key=lambda x: x['importance'], reverse=True)
        
#         # First pass: identify main topics and potential subtopics
#         for i, topic in enumerate(sorted_topics):
#             is_subtopic = False
            
#             for j, potential_parent in enumerate(sorted_topics):
#                 if i != j and similarity_matrix[i][j] > 0.5 and topic['importance'] < potential_parent['importance']:
#                     # This topic is similar to a more important topic - mark as potential subtopic
#                     is_subtopic = True
#                     break
            
#             if not is_subtopic:
#                 main_topics.append(topic)
#             else:
#                 subtopics.add(topic['id'])
        
#         # Second pass: assign subtopics to main topics
#         for main_topic in main_topics:
#             hierarchy[main_topic['id']] = []
            
#             for i, topic in enumerate(sorted_topics):
#                 if topic['id'] in subtopics:  # Only consider topics marked as subtopics
#                     j = next((idx for idx, t in enumerate(sorted_topics) if t['id'] == main_topic['id']), None)
                    
#                     if j is not None and similarity_matrix[i][j] > 0.35:  # Threshold for parent-child similarity
#                         hierarchy[main_topic['id']].append(topic['id'])
#         print('heir')
#         return hierarchy, main_topics, [t for t in topics if t['id'] in subtopics]

#     def create_knowledge_graph(self, text):
#         # Extract topics
#         topics = self.extract_topics_and_subtopics(text)
        
#         # Create hierarchical structure
#         hierarchy, main_topics, subtopics = self.create_hierarchical_topic_structure(topics)
        
#         # Initialize graph
#         nodes = []
#         edges = []
        
#         # Add document root node
#         root_id = 'doc_root'
#         nodes.append({
#             'id': root_id,
#             'label': 'Document',
#             'type': 'DOCUMENT',
#             'importance': 10
#         })
        
#         # Process topics
#         all_topic_ids = set()
#         topic_map = {topic['id']: topic for topic in topics}
        
#         # Add main topics connected to root
#         for topic in main_topics:
#             topic_id = topic['id']
#             all_topic_ids.add(topic_id)
            
#             # Add node
#             nodes.append({
#                 'id': topic_id,
#                 'label': topic['label'],
#                 'type': 'MAIN_TOPIC',
#                 'importance': topic['importance'],
#                 'terms': topic['terms'][:5]  # Store top 5 terms for reference
#             })
            
#             # Connect to root
#             edges.append({
#                 'source': root_id,
#                 'target': topic_id,
#                 'relationship': 'CONTAINS',
#                 'weight': topic['importance']
#             })
            
#             # Add subtopics if present
#             if topic_id in hierarchy and hierarchy[topic_id]:
#                 for subtopic_id in hierarchy[topic_id]:
#                     if subtopic_id in topic_map and subtopic_id not in all_topic_ids:
#                         subtopic = topic_map[subtopic_id]
#                         all_topic_ids.add(subtopic_id)
                        
#                         # Add subtopic node
#                         nodes.append({
#                             'id': subtopic_id,
#                             'label': subtopic['label'],
#                             'type': 'SUBTOPIC',
#                             'importance': subtopic['importance'] * 0.8,  # Slightly smaller than main topics
#                             'terms': subtopic['terms'][:3]  # Store top 3 terms
#                         })
                        
#                         # Connect to main topic
#                         edges.append({
#                             'source': topic_id,
#                             'target': subtopic_id,
#                             'relationship': 'HAS_SUBTOPIC',
#                             'weight': subtopic['importance']
#                         })
            
#             # Extract entities and concepts for this topic
#             topic_sentences = topic['sentences']
#             entities, concepts = self.extract_entities_and_concepts(topic_sentences)
            
#             # Add important entities
#             entity_count = 0
#             for key, entity in sorted(entities.items(), key=lambda x: x[1]['count'], reverse=True):
#                 if entity_count >= 5:  # Limit to top 5 entities per topic
#                     break
                    
#                 if entity['count'] >= 2:  # Must appear at least twice
#                     entity_id = f"entity_{topic_id}_{entity_count}"
                    
#                     # Add entity node
#                     nodes.append({
#                         'id': entity_id,
#                         'label': entity['text'],
#                         'type': 'ENTITY',
#                         'entity_type': entity['type'],
#                         'importance': min(8, entity['count'] * 1.5)
#                     })
                    
#                     # Connect to topic
#                     edges.append({
#                         'source': topic_id,
#                         'target': entity_id,
#                         'relationship': 'HAS_ENTITY',
#                         'weight': min(5, entity['count'])
#                     })
                    
#                     entity_count += 1
            
#             # Add important concepts
#             concept_count = 0
#             for key, concept in sorted(concepts.items(), key=lambda x: x[1]['count'], reverse=True):
#                 if concept_count >= 7:  # Limit to top 7 concepts per topic
#                     break
                    
#                 if concept['count'] >= 2:  # Must appear at least twice
#                     concept_id = f"concept_{topic_id}_{concept_count}"
                    
#                     # Add concept node
#                     nodes.append({
#                         'id': concept_id,
#                         'label': concept['text'].title(),
#                         'type': 'CONCEPT',
#                         'importance': min(6, concept['count'])
#                     })
                    
#                     # Connect to topic
#                     edges.append({
#                         'source': topic_id,
#                         'target': concept_id,
#                         'relationship': 'HAS_CONCEPT',
#                         'weight': min(4, concept['count'])
#                     })
                    
#                     concept_count += 1
        
#         # Extract cross-topic relationships
#         relationships = self.extract_relationships([s for t in topics for s in t['sentences']])
        
#         # Add significant relationships between entities and concepts
#         added_relationships = set()  # To avoid duplicates
        
#         for rel in relationships:
#             source_text = rel['source'].lower()
#             target_text = rel['target'].lower()
            
#             # Find nodes with matching labels
#             source_nodes = [n for n in nodes if n['label'].lower() == source_text and n['type'] in ['ENTITY', 'CONCEPT']]
#             target_nodes = [n for n in nodes if n['label'].lower() == target_text and n['type'] in ['ENTITY', 'CONCEPT']]
            
#             for source_node in source_nodes:
#                 for target_node in target_nodes:
#                     rel_key = f"{source_node['id']}_{target_node['id']}"
                    
#                     if rel_key not in added_relationships:
#                         edges.append({
#                             'source': source_node['id'],
#                             'target': target_node['id'],
#                             'relationship': rel['relationship'].upper(),
#                             'weight': 3
#                         })
#                         added_relationships.add(rel_key)
        
#         return {
#             'nodes': nodes,
#             'edges': edges,
#             'metadata': {
#                 'topic_count': len(main_topics),
#                 'subtopic_count': len(subtopics),
#                 'entity_count': len([n for n in nodes if n['type'] == 'ENTITY']),
#                 'concept_count': len([n for n in nodes if n['type'] == 'CONCEPT']),
#                 'relationship_count': len(edges)
#             }
#         }

# def generate_enhanced_knowledge_graph(file_path):
#     """Generate enhanced knowledge graph from PDF content."""
#     try:
#         text = extract_text_from_pdf(file_path)
#         extractor = EnhancedKnowledgeGraphExtractor()
#         graph = extractor.create_knowledge_graph(text)
#         return graph
#     except Exception as e:
#         print(f"Error generating knowledge graph: {str(e)}")
#         return None


# from spacy import load
# import networkx as nx
# from textblob import TextBlob
# from app.services.pdf_processing import extract_text_from_pdf
# import spacy
# from collections import defaultdict
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# import numpy as np

# class KnowledgeGraphExtractor:
#     def __init__(self):
#         self.nlp = spacy.load("en_core_web_sm")
        
#     def extract_topics_and_subtopics(self, text):
#         # Split text into sentences
#         sentences = [sent.text.strip() for sent in self.nlp(text).sents]
        
#         # Calculate number of topics based on text length
#         n_clusters = min(max(3, len(sentences) // 20), 8)  # 1 topic per ~20 sentences, min 3, max 8
        
#         # Create TF-IDF vectorizer with more features for longer texts
#         max_features = min(100 + (len(sentences) // 10), 300)  # Scale features with text length
#         vectorizer = TfidfVectorizer(
#             max_features=max_features, 
#             stop_words='english',
#             ngram_range=(1, 2)
#         )
#         tfidf_matrix = vectorizer.fit_transform(sentences)
        
#         # Cluster sentences into topics
#         kmeans = KMeans(n_clusters=n_clusters)
#         clusters = kmeans.fit_predict(tfidf_matrix)
        
#         # Get important terms for each cluster
#         terms = vectorizer.get_feature_names_out()
#         centroids = kmeans.cluster_centers_
        
#         # Calculate cluster sizes for importance
#         cluster_sizes = [sum(clusters == i) for i in range(n_clusters)]
#         max_cluster_size = max(cluster_sizes)
        
#         topics = []
#         for i in range(n_clusters):
#             # Scale importance based on cluster size
#             importance = (cluster_sizes[i] / max_cluster_size) * 10
            
#             # Get top terms for this cluster
#             top_term_indices = centroids[i].argsort()[-5:][::-1]
#             topic_terms = [terms[idx] for idx in top_term_indices]
            
#             topics.append({
#                 'id': f'topic_{i}',
#                 'label': ' '.join(topic_terms[:2]),
#                 'type': 'MAIN_TOPIC',
#                 'sentences': [sent for j, sent in enumerate(sentences) if clusters[j] == i],
#                 'terms': topic_terms,
#                 'importance': importance
#             })
        
#         return topics

#     def create_knowledge_graph(self, text):
#         doc = self.nlp(text)
        
#         # Track unique entities and concepts
#         unique_nodes = {}
#         edges = []
        
#         # Add document root node
#         root_id = 'doc_root'
#         unique_nodes[root_id] = {
#             'id': root_id,
#             'label': 'Document',
#             'type': 'DOCUMENT'
#         }
        
#         # Extract topics
#         topics = self.extract_topics_and_subtopics(text)
        
#         # Process each topic
#         for topic in topics:
#             topic_id = topic['id']
#             if topic['label'] not in [node['label'] for node in unique_nodes.values()]:
#                 unique_nodes[topic_id] = {
#                     'id': topic_id,
#                     'label': topic['label'],
#                     'type': 'MAIN_TOPIC'
#                 }
                
#                 # Connect to root
#                 edges.append({
#                     'source': root_id,
#                     'target': topic_id,
#                     'relationship': 'CONTAINS'
#                 })
                
#                 # Process topic content
#                 topic_doc = self.nlp(' '.join(topic['sentences']))
                
#                 # Track entities and concepts for this topic
#                 topic_elements = defaultdict(int)
                
#                 # Extract entities
#                 for ent in topic_doc.ents:
#                     if len(ent.text.strip()) > 2:
#                         topic_elements[ent.text.lower()] += 1
                
#                 # Extract noun phrases
#                 for chunk in topic_doc.noun_chunks:
#                     if len(chunk.text.strip()) > 2:
#                         topic_elements[chunk.text.lower()] += 1
                
#                 # Scale importance based on counts
#                 max_entity_count = max(topic_elements.values()) if topic_elements else 1
#                 max_concept_count = max(topic_elements.values()) if topic_elements else 1
                
#                 # Add entities and concepts with scaled importance
#                 for ent, count in topic_elements.items():
#                     importance = (count / max_entity_count) * 8  # Scale to reasonable size
#                     element_id = f"element_{len(unique_nodes)}"
#                     if ent not in [node['label'].lower() for node in unique_nodes.values()]:
#                         unique_nodes[element_id] = {
#                             'id': element_id,
#                             'label': ent.title(),
#                             'type': 'CONCEPT',
#                             'importance': importance
#                         }
                        
#                         edges.append({
#                             'source': topic_id,
#                             'target': element_id,
#                             'relationship': 'CONTAINS',
#                             'weight': importance
#                         })
        
#         return {
#             'nodes': list(unique_nodes.values()),
#             'edges': edges
#         }

# def generate_knowledge_graph(file_path):
#     """Generate knowledge graph from PDF content."""
#     try:
#         text = extract_text_from_pdf(file_path)
#         extractor = KnowledgeGraphExtractor()
#         graph = extractor.create_knowledge_graph(text)
#         return graph
#     except Exception as e:
#         print(f"Error generating knowledge graph: {str(e)}")
#         return None 
