import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.core.embeddings import EmbeddingProvider, EmbeddingManager
from src.models.database import SessionLocal, EmbeddingEvaluation, EmbeddingModel
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TestQuery:
    """A test query for evaluation."""
    id: str
    text: str
    expected_documents: List[str] = field(default_factory=list)
    expected_sections: List[str] = field(default_factory=list)
    query_type: str = "general"  # general, specific, cross_reference
    difficulty: str = "medium"   # easy, medium, hard
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a single retrieval query."""
    query_id: str
    retrieved_docs: List[str]
    scores: List[float]
    expected_docs: List[str]
    precision_at_5: float
    recall_at_10: float
    reciprocal_rank: float
    processing_time_ms: int


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""
    # Retrieval metrics
    avg_precision_at_5: float
    avg_recall_at_10: float
    avg_reciprocal_rank: float
    
    # Semantic clustering metrics
    silhouette_score: float
    intra_cluster_similarity: float
    inter_cluster_separation: float
    
    # Cross-document retrieval
    cross_document_precision: float
    cross_document_recall: float
    
    # Performance metrics
    avg_embedding_time_ms: float
    avg_search_time_ms: float
    embeddings_per_second: float
    
    # Cost metrics
    total_cost_usd: float
    cost_per_query: float
    cost_per_1k_tokens: float
    
    # Quality scores
    overall_quality_score: float
    retrieval_quality: float
    semantic_quality: float
    
    # Detailed results
    query_results: List[RetrievalResult] = field(default_factory=list)
    clustering_details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyQueryGenerator:
    """Generates test queries specific to policy documents."""
    
    def __init__(self):
        self.policy_acronyms = ["ADM", "CARE", "EPM", "EVS", "IPC", "LEG", "MAINT", "PRV", "RC"]
        self.document_types = ["policy", "procedure", "form", "checklist", "guide"]
    
    def generate_test_queries(self) -> List[TestQuery]:
        """Generate comprehensive test queries for policy evaluation."""
        queries = []
        
        # 1. Specific policy queries
        queries.extend(self._generate_specific_policy_queries())
        
        # 2. Cross-reference queries
        queries.extend(self._generate_cross_reference_queries())
        
        # 3. Procedural queries
        queries.extend(self._generate_procedural_queries())
        
        # 4. Compliance queries
        queries.extend(self._generate_compliance_queries())
        
        # 5. Emergency and safety queries
        queries.extend(self._generate_safety_queries())
        
        return queries
    
    def _generate_specific_policy_queries(self) -> List[TestQuery]:
        """Generate queries about specific policies."""
        return [
            TestQuery(
                id="infection_control_01",
                text="What are the hand hygiene requirements for staff?",
                expected_sections=["IPC"],
                query_type="specific",
                difficulty="easy"
            ),
            TestQuery(
                id="medication_management_01",
                text="How should controlled substances be stored and monitored?",
                expected_sections=["CARE"],
                query_type="specific",
                difficulty="medium"
            ),
            TestQuery(
                id="emergency_procedures_01",
                text="What is the fire evacuation procedure for residents with mobility issues?",
                expected_sections=["EPM"],
                query_type="specific",
                difficulty="medium"
            ),
            TestQuery(
                id="privacy_confidentiality_01",
                text="What information can be shared with family members without consent?",
                expected_sections=["PRV"],
                query_type="specific",
                difficulty="hard"
            ),
            TestQuery(
                id="maintenance_safety_01",
                text="How often should medical equipment be inspected and calibrated?",
                expected_sections=["MAINT"],
                query_type="specific",
                difficulty="medium"
            )
        ]
    
    def _generate_cross_reference_queries(self) -> List[TestQuery]:
        """Generate queries that span multiple policy areas."""
        return [
            TestQuery(
                id="cross_infection_care_01",
                text="What infection control measures are required during personal care?",
                expected_sections=["IPC", "CARE"],
                query_type="cross_reference",
                difficulty="medium"
            ),
            TestQuery(
                id="cross_emergency_care_01",
                text="How do emergency procedures change for residents on isolation precautions?",
                expected_sections=["EPM", "IPC"],
                query_type="cross_reference",
                difficulty="hard"
            ),
            TestQuery(
                id="cross_privacy_admin_01",
                text="What documentation is required for privacy breach incidents?",
                expected_sections=["PRV", "ADM"],
                query_type="cross_reference",
                difficulty="medium"
            )
        ]
    
    def _generate_procedural_queries(self) -> List[TestQuery]:
        """Generate queries about procedures and workflows."""
        return [
            TestQuery(
                id="procedure_admission_01",
                text="What steps are involved in the resident admission process?",
                expected_sections=["ADM", "CARE"],
                query_type="procedural",
                difficulty="medium"
            ),
            TestQuery(
                id="procedure_incident_01",
                text="How should staff respond to a resident fall incident?",
                expected_sections=["CARE", "ADM"],
                query_type="procedural",
                difficulty="medium"
            ),
            TestQuery(
                id="procedure_discharge_01",
                text="What documentation is required for resident discharge?",
                expected_sections=["ADM", "PRV"],
                query_type="procedural",
                difficulty="medium"
            )
        ]
    
    def _generate_compliance_queries(self) -> List[TestQuery]:
        """Generate compliance and regulatory queries."""
        return [
            TestQuery(
                id="compliance_training_01",
                text="What training is required for new nursing staff?",
                expected_sections=["ADM", "CARE"],
                query_type="compliance",
                difficulty="medium"
            ),
            TestQuery(
                id="compliance_documentation_01",
                text="How long must resident care records be retained?",
                expected_sections=["ADM", "PRV"],
                query_type="compliance",
                difficulty="hard"
            ),
            TestQuery(
                id="compliance_reporting_01",
                text="What incidents must be reported to regulatory authorities?",
                expected_sections=["ADM", "LEG"],
                query_type="compliance",
                difficulty="hard"
            )
        ]
    
    def _generate_safety_queries(self) -> List[TestQuery]:
        """Generate safety and emergency queries."""
        return [
            TestQuery(
                id="safety_chemical_01",
                text="How should cleaning chemicals be stored and labeled?",
                expected_sections=["EVS", "MAINT"],
                query_type="safety",
                difficulty="easy"
            ),
            TestQuery(
                id="safety_lift_01",
                text="What safety procedures apply when using mechanical lifts?",
                expected_sections=["CARE", "MAINT"],
                query_type="safety",
                difficulty="medium"
            ),
            TestQuery(
                id="safety_outbreak_01",
                text="What steps should be taken during an infectious disease outbreak?",
                expected_sections=["IPC", "EPM"],
                query_type="safety",
                difficulty="hard"
            )
        ]


class EmbeddingEvaluator:
    """Comprehensive embedding evaluation framework."""
    
    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.query_generator = PolicyQueryGenerator()
    
    def evaluate_provider(
        self,
        provider_name: str,
        test_documents: Optional[List[Dict[str, Any]]] = None,
        custom_queries: Optional[List[TestQuery]] = None,
        save_results: bool = True
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of an embedding provider.
        
        Args:
            provider_name: Name of the provider to evaluate
            test_documents: Documents to use for evaluation (from database if None)
            custom_queries: Custom test queries (generated if None)
            save_results: Whether to save results to database
            
        Returns:
            Comprehensive evaluation metrics
        """
        logger.info(f"Starting evaluation of provider: {provider_name}")
        start_time = time.time()
        
        # Get provider
        provider = self.embedding_manager.get_provider(provider_name)
        
        # Get test documents
        if test_documents is None:
            test_documents = self._get_test_documents()
        
        # Get test queries
        test_queries = custom_queries or self.query_generator.generate_test_queries()
        
        # Embed all documents
        logger.info(f"Embedding {len(test_documents)} test documents...")
        doc_embeddings = self._embed_documents(provider, test_documents)
        
        # Embed all queries
        logger.info(f"Embedding {len(test_queries)} test queries...")
        query_embeddings = self._embed_queries(provider, test_queries)
        
        # Evaluate retrieval performance
        logger.info("Evaluating retrieval performance...")
        retrieval_metrics = self._evaluate_retrieval(
            test_queries, query_embeddings, test_documents, doc_embeddings
        )
        
        # Evaluate semantic clustering
        logger.info("Evaluating semantic clustering...")
        clustering_metrics = self._evaluate_clustering(test_documents, doc_embeddings)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            provider, len(test_documents), len(test_queries)
        )
        
        # Calculate cost metrics
        cost_metrics = self._calculate_cost_metrics(
            provider, len(test_documents), len(test_queries)
        )
        
        # Combine all metrics
        evaluation_metrics = EvaluationMetrics(
            # Retrieval metrics
            avg_precision_at_5=np.mean([r.precision_at_5 for r in retrieval_metrics]),
            avg_recall_at_10=np.mean([r.recall_at_10 for r in retrieval_metrics]),
            avg_reciprocal_rank=np.mean([r.reciprocal_rank for r in retrieval_metrics]),
            
            # Clustering metrics
            silhouette_score=clustering_metrics["silhouette_score"],
            intra_cluster_similarity=clustering_metrics["intra_cluster_similarity"],
            inter_cluster_separation=clustering_metrics["inter_cluster_separation"],
            
            # Cross-document metrics
            cross_document_precision=self._calculate_cross_document_precision(retrieval_metrics),
            cross_document_recall=self._calculate_cross_document_recall(retrieval_metrics),
            
            # Performance metrics
            avg_embedding_time_ms=performance_metrics["avg_embedding_time_ms"],
            avg_search_time_ms=performance_metrics["avg_search_time_ms"],
            embeddings_per_second=performance_metrics["embeddings_per_second"],
            
            # Cost metrics
            total_cost_usd=cost_metrics["total_cost"],
            cost_per_query=cost_metrics["cost_per_query"],
            cost_per_1k_tokens=provider.cost_per_1k_tokens or 0.0,
            
            # Quality scores
            overall_quality_score=0.0,  # Will calculate below
            retrieval_quality=0.0,      # Will calculate below
            semantic_quality=0.0,       # Will calculate below
            
            # Detailed results
            query_results=retrieval_metrics,
            clustering_details=clustering_metrics,
            metadata={
                "provider_name": provider_name,
                "model_name": provider.model_name,
                "evaluation_date": datetime.utcnow().isoformat(),
                "test_document_count": len(test_documents),
                "test_query_count": len(test_queries),
                "total_evaluation_time_ms": int((time.time() - start_time) * 1000)
            }
        )
        
        # Calculate quality scores
        evaluation_metrics.retrieval_quality = self._calculate_retrieval_quality(evaluation_metrics)
        evaluation_metrics.semantic_quality = self._calculate_semantic_quality(evaluation_metrics)
        evaluation_metrics.overall_quality_score = self._calculate_overall_quality(evaluation_metrics)
        
        logger.info(f"Evaluation complete. Overall quality score: {evaluation_metrics.overall_quality_score:.3f}")
        
        # Save results to database
        if save_results:
            self._save_evaluation_results(provider_name, evaluation_metrics)
        
        return evaluation_metrics
    
    def _get_test_documents(self) -> List[Dict[str, Any]]:
        """Get test documents from database."""
        db = SessionLocal()
        try:
            from src.models.database import Document, DocumentChunk
            
            # Get sample documents with chunks
            documents = db.query(Document).filter(
                Document.parsing_success == True,
                Document.total_chunks > 0
            ).limit(100).all()  # Limit for evaluation
            
            test_docs = []
            for doc in documents:
                chunks = db.query(DocumentChunk).filter_by(document_id=doc.id).limit(5).all()
                
                for chunk in chunks:
                    test_docs.append({
                        "id": f"{doc.id}_{chunk.id}",
                        "text": chunk.chunk_text,
                        "document_id": doc.id,
                        "chunk_id": chunk.id,
                        "document_index": doc.document_index,
                        "policy_acronym": chunk.context_metadata.get("policy_acronym"),
                        "section": chunk.context_metadata.get("section"),
                        "document_type": chunk.context_metadata.get("document_type"),
                        "metadata": chunk.context_metadata
                    })
            
            logger.info(f"Retrieved {len(test_docs)} test documents from database")
            return test_docs
            
        finally:
            db.close()
    
    def _embed_documents(self, provider: EmbeddingProvider, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Embed all test documents."""
        texts = [doc["text"] for doc in documents]
        result = provider.embed_texts(texts)
        return np.array(result.embeddings)
    
    def _embed_queries(self, provider: EmbeddingProvider, queries: List[TestQuery]) -> np.ndarray:
        """Embed all test queries."""
        texts = [query.text for query in queries]
        result = provider.embed_texts(texts)
        return np.array(result.embeddings)
    
    def _evaluate_retrieval(
        self,
        queries: List[TestQuery],
        query_embeddings: np.ndarray,
        documents: List[Dict[str, Any]],
        doc_embeddings: np.ndarray
    ) -> List[RetrievalResult]:
        """Evaluate retrieval performance."""
        results = []
        
        for i, query in enumerate(queries):
            start_time = time.time()
            
            # Calculate similarities
            query_embedding = query_embeddings[i:i+1]
            similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:20]  # Top 20
            retrieved_docs = [documents[idx]["id"] for idx in top_indices]
            scores = [similarities[idx] for idx in top_indices]
            
            # Calculate metrics
            precision_at_5 = self._calculate_precision_at_k(retrieved_docs[:5], query.expected_documents)
            recall_at_10 = self._calculate_recall_at_k(retrieved_docs[:10], query.expected_documents)
            reciprocal_rank = self._calculate_reciprocal_rank(retrieved_docs, query.expected_documents)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            results.append(RetrievalResult(
                query_id=query.id,
                retrieved_docs=retrieved_docs,
                scores=scores,
                expected_docs=query.expected_documents,
                precision_at_5=precision_at_5,
                recall_at_10=recall_at_10,
                reciprocal_rank=reciprocal_rank,
                processing_time_ms=processing_time
            ))
        
        return results
    
    def _evaluate_clustering(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, Any]:
        """Evaluate semantic clustering quality."""
        try:
            # Group documents by policy acronym
            policy_groups = {}
            for i, doc in enumerate(documents):
                policy = doc.get("policy_acronym", "unknown")
                if policy not in policy_groups:
                    policy_groups[policy] = []
                policy_groups[policy].append(i)
            
            # Calculate silhouette score
            if len(policy_groups) > 1:
                labels = []
                for i, doc in enumerate(documents):
                    policy = doc.get("policy_acronym", "unknown")
                    labels.append(list(policy_groups.keys()).index(policy))
                
                silhouette = silhouette_score(embeddings, labels)
            else:
                silhouette = 0.0
            
            # Calculate intra-cluster similarity
            intra_similarities = []
            for policy, indices in policy_groups.items():
                if len(indices) > 1:
                    group_embeddings = embeddings[indices]
                    similarities = cosine_similarity(group_embeddings)
                    # Average similarity within group (excluding diagonal)
                    mask = np.ones_like(similarities, dtype=bool)
                    np.fill_diagonal(mask, False)
                    intra_similarities.append(similarities[mask].mean())
            
            avg_intra_similarity = np.mean(intra_similarities) if intra_similarities else 0.0
            
            # Calculate inter-cluster separation
            inter_similarities = []
            policy_list = list(policy_groups.keys())
            for i in range(len(policy_list)):
                for j in range(i+1, len(policy_list)):
                    group1_embeddings = embeddings[policy_groups[policy_list[i]]]
                    group2_embeddings = embeddings[policy_groups[policy_list[j]]]
                    
                    similarities = cosine_similarity(group1_embeddings, group2_embeddings)
                    inter_similarities.append(similarities.mean())
            
            avg_inter_separation = 1.0 - np.mean(inter_similarities) if inter_similarities else 0.0
            
            return {
                "silhouette_score": silhouette,
                "intra_cluster_similarity": avg_intra_similarity,
                "inter_cluster_separation": avg_inter_separation,
                "policy_group_count": len(policy_groups),
                "policy_groups": {k: len(v) for k, v in policy_groups.items()}
            }
            
        except Exception as e:
            logger.warning(f"Error in clustering evaluation: {e}")
            return {
                "silhouette_score": 0.0,
                "intra_cluster_similarity": 0.0,
                "inter_cluster_separation": 0.0,
                "error": str(e)
            }
    
    def _calculate_precision_at_k(self, retrieved: List[str], expected: List[str], k: int = 5) -> float:
        """Calculate precision@k."""
        if not expected or not retrieved:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in top_k if doc in expected)
        return relevant_retrieved / len(top_k)
    
    def _calculate_recall_at_k(self, retrieved: List[str], expected: List[str], k: int = 10) -> float:
        """Calculate recall@k."""
        if not expected:
            return 0.0
        
        top_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc in top_k if doc in expected)
        return relevant_retrieved / len(expected)
    
    def _calculate_reciprocal_rank(self, retrieved: List[str], expected: List[str]) -> float:
        """Calculate reciprocal rank."""
        for i, doc in enumerate(retrieved):
            if doc in expected:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_cross_document_precision(self, results: List[RetrievalResult]) -> float:
        """Calculate precision for cross-document queries."""
        cross_results = [r for r in results if "_cross_" in r.query_id]
        if not cross_results:
            return 0.0
        return np.mean([r.precision_at_5 for r in cross_results])
    
    def _calculate_cross_document_recall(self, results: List[RetrievalResult]) -> float:
        """Calculate recall for cross-document queries."""
        cross_results = [r for r in results if "_cross_" in r.query_id]
        if not cross_results:
            return 0.0
        return np.mean([r.recall_at_10 for r in cross_results])
    
    def _calculate_performance_metrics(self, provider: EmbeddingProvider, doc_count: int, query_count: int) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Simple estimation based on typical embedding times
        model_speed_estimates = {
            "sentence_transformers": 100,  # embeddings per second
            "openai": 50,                  # embeddings per second (API limited)
            "cohere": 30                   # embeddings per second (API limited)
        }
        
        speed = model_speed_estimates.get(provider.provider_type, 50)
        avg_embedding_time = 1000 / speed  # ms per embedding
        avg_search_time = 1.0  # ms per search (cosine similarity is fast)
        
        return {
            "avg_embedding_time_ms": avg_embedding_time,
            "avg_search_time_ms": avg_search_time,
            "embeddings_per_second": speed
        }
    
    def _calculate_cost_metrics(self, provider: EmbeddingProvider, doc_count: int, query_count: int) -> Dict[str, float]:
        """Calculate cost metrics."""
        if not provider.cost_per_1k_tokens:
            return {"total_cost": 0.0, "cost_per_query": 0.0}
        
        # Estimate tokens (rough approximation)
        avg_doc_length = 200  # words
        avg_query_length = 10  # words
        total_tokens = (doc_count * avg_doc_length + query_count * avg_query_length) * 0.75  # words to tokens
        
        total_cost = (total_tokens / 1000) * provider.cost_per_1k_tokens
        cost_per_query = total_cost / query_count if query_count > 0 else 0.0
        
        return {
            "total_cost": total_cost,
            "cost_per_query": cost_per_query
        }
    
    def _calculate_retrieval_quality(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall retrieval quality score."""
        # Weighted combination of retrieval metrics
        weights = {
            "precision": 0.4,
            "recall": 0.3,
            "reciprocal_rank": 0.3
        }
        
        quality = (
            weights["precision"] * metrics.avg_precision_at_5 +
            weights["recall"] * metrics.avg_recall_at_10 +
            weights["reciprocal_rank"] * metrics.avg_reciprocal_rank
        )
        
        return min(1.0, max(0.0, quality))
    
    def _calculate_semantic_quality(self, metrics: EvaluationMetrics) -> float:
        """Calculate semantic clustering quality score."""
        # Weighted combination of clustering metrics
        weights = {
            "silhouette": 0.4,
            "intra_similarity": 0.3,
            "inter_separation": 0.3
        }
        
        # Normalize silhouette score from [-1, 1] to [0, 1]
        normalized_silhouette = (metrics.silhouette_score + 1) / 2
        
        quality = (
            weights["silhouette"] * normalized_silhouette +
            weights["intra_similarity"] * metrics.intra_cluster_similarity +
            weights["inter_separation"] * metrics.inter_cluster_separation
        )
        
        return min(1.0, max(0.0, quality))
    
    def _calculate_overall_quality(self, metrics: EvaluationMetrics) -> float:
        """Calculate overall quality score."""
        # Weighted combination of all quality aspects
        weights = {
            "retrieval": 0.5,
            "semantic": 0.3,
            "cross_document": 0.2
        }
        
        cross_document_quality = (metrics.cross_document_precision + metrics.cross_document_recall) / 2
        
        overall = (
            weights["retrieval"] * metrics.retrieval_quality +
            weights["semantic"] * metrics.semantic_quality +
            weights["cross_document"] * cross_document_quality
        )
        
        return min(1.0, max(0.0, overall))
    
    def _save_evaluation_results(self, provider_name: str, metrics: EvaluationMetrics):
        """Save evaluation results to database."""
        db = SessionLocal()
        try:
            # Find or create embedding model record
            embedding_model = db.query(EmbeddingModel).filter_by(
                provider=provider_name.split("_")[0],  # Remove variant suffix
                model_name=metrics.metadata["model_name"]
            ).first()
            
            if not embedding_model:
                provider_info = self.embedding_manager.get_provider_info(provider_name)
                embedding_model = EmbeddingModel(
                    provider=provider_name.split("_")[0],
                    model_name=metrics.metadata["model_name"],
                    dimension=provider_info["dimension"],
                    cost_per_1k_tokens=provider_info.get("cost_per_1k_tokens"),
                    is_active=True
                )
                db.add(embedding_model)
                db.commit()
                db.refresh(embedding_model)
            
            # Create evaluation record
            evaluation = EmbeddingEvaluation(
                evaluation_name=f"{provider_name}_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                embedding_model_id=embedding_model.id,
                test_query_count=len(metrics.query_results),
                evaluation_date=datetime.utcnow(),
                precision_at_5=metrics.avg_precision_at_5,
                recall_at_10=metrics.avg_recall_at_10,
                avg_response_time_ms=metrics.avg_search_time_ms,
                semantic_clustering_score=metrics.silhouette_score,
                cross_document_score=(metrics.cross_document_precision + metrics.cross_document_recall) / 2,
                total_cost_usd=metrics.total_cost_usd,
                cost_per_query=metrics.cost_per_query,
                detailed_results=metrics.metadata
            )
            
            db.add(evaluation)
            db.commit()
            
            logger.info(f"Saved evaluation results for {provider_name}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            db.rollback()
        finally:
            db.close()


# Global evaluator instance
embedding_evaluator = EmbeddingEvaluator()