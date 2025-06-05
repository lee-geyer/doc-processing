#!/usr/bin/env python3
"""
Collection Manager for policyQA naming convention.
Supports multiple embedding models and easy comparison.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.core.vector_store import PolicyVectorStore
from src.core.embeddings import embedding_manager, ProviderConfig
import requests

console = Console()

@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    name: str
    provider_type: str
    model_name: str
    dimension: int
    description: str
    cost_per_1k_tokens: float = 0.0
    batch_size: int = 32

class PolicyQACollectionManager:
    """Manage policyQA collections with clear naming convention."""
    
    def __init__(self):
        self.vector_store = PolicyVectorStore()
        self.models = self._get_model_configs()
    
    def _get_model_configs(self) -> Dict[str, ModelConfig]:
        """Get all available model configurations."""
        return {
            # Dense embedding models
            "dense_mpnet768": ModelConfig(
                name="policyQA_dense_mpnet768",
                provider_type="sentence_transformers",
                model_name="all-mpnet-base-v2",
                dimension=768,
                description="General purpose, balanced (current baseline)",
                cost_per_1k_tokens=0.0,
                batch_size=32
            ),
            "dense_qa768": ModelConfig(
                name="policyQA_dense_qa768",
                provider_type="sentence_transformers", 
                model_name="multi-qa-mpnet-base-dot-v1",
                dimension=768,
                description="Q&A optimized, same dimension as baseline",
                cost_per_1k_tokens=0.0,
                batch_size=32
            ),
            "dense_mini384": ModelConfig(
                name="policyQA_dense_mini384",
                provider_type="sentence_transformers",
                model_name="all-MiniLM-L6-v2", 
                dimension=384,
                description="Fast inference, smaller vectors",
                cost_per_1k_tokens=0.0,
                batch_size=64
            ),
            "dense_openai1536": ModelConfig(
                name="policyQA_dense_openai1536",
                provider_type="openai",
                model_name="text-embedding-3-small",
                dimension=1536,
                description="OpenAI small model, good quality",
                cost_per_1k_tokens=0.00002,
                batch_size=100
            ),
            "dense_openai3072": ModelConfig(
                name="policyQA_dense_openai3072", 
                provider_type="openai",
                model_name="text-embedding-3-large",
                dimension=3072,
                description="OpenAI large model, highest quality",
                cost_per_1k_tokens=0.00013,
                batch_size=100
            ),
            "dense_cohere1024": ModelConfig(
                name="policyQA_dense_cohere1024",
                provider_type="cohere",
                model_name="embed-english-v3.0",
                dimension=1024,
                description="Cohere English model, competitive",
                cost_per_1k_tokens=0.0001,
                batch_size=96
            )
        }
    
    def list_available_models(self):
        """List all available models."""
        console.print("[bold cyan]Available policyQA Models[/bold cyan]\n")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Collection Name", style="cyan", width=30)
        table.add_column("Model", style="green", width=35)
        table.add_column("Dim", justify="right", style="yellow", width=5)
        table.add_column("Cost/1K", justify="right", style="red", width=8)
        table.add_column("Description", style="dim")
        
        for config in self.models.values():
            cost_str = "Free" if config.cost_per_1k_tokens == 0 else f"${config.cost_per_1k_tokens:.5f}"
            table.add_row(
                config.name,
                config.model_name,
                str(config.dimension),
                cost_str,
                config.description
            )
        
        console.print(table)
    
    def list_existing_collections(self):
        """List existing collections in Qdrant."""
        try:
            response = requests.get('http://localhost:6333/collections', timeout=5)
            data = response.json()
            collections = data['result']['collections']
            
            console.print(f"\n[bold yellow]Existing Collections ({len(collections)})[/bold yellow]\n")
            
            if not collections:
                console.print("[dim]No collections found[/dim]")
                return
            
            table = Table(show_header=True)
            table.add_column("Collection Name", style="cyan")
            table.add_column("Vectors", justify="right", style="green")
            table.add_column("Status", style="yellow")
            
            for collection in collections:
                name = collection['name']
                
                # Get collection info
                try:
                    info_response = requests.get(f'http://localhost:6333/collections/{name}', timeout=5)
                    info_data = info_response.json()
                    vector_count = info_data['result']['points_count']
                    status = "✓ Ready"
                except:
                    vector_count = "Unknown"
                    status = "⚠ Error"
                
                table.add_row(name, str(vector_count), status)
            
            console.print(table)
            
        except Exception as e:
            console.print(f"[red]Error checking collections: {e}[/red]")
    
    def create_collection(self, model_key: str, force: bool = False) -> bool:
        """Create a collection for the specified model."""
        if model_key not in self.models:
            console.print(f"[red]Unknown model: {model_key}[/red]")
            console.print(f"Available: {', '.join(self.models.keys())}")
            return False
        
        config = self.models[model_key]
        console.print(f"[cyan]Creating collection: {config.name}[/cyan]")
        
        # Check if collection exists
        if self.vector_store.collection_exists(config.name):
            if not force:
                console.print(f"[yellow]Collection {config.name} already exists. Use --force to recreate.[/yellow]")
                return False
            else:
                console.print(f"[yellow]Deleting existing collection: {config.name}[/yellow]")
                self._delete_collection(config.name)
        
        # Create embedding provider
        provider_config = ProviderConfig(
            provider_type=config.provider_type,
            model_name=config.model_name,
            dimension=config.dimension,
            batch_size=config.batch_size,
            cost_per_1k_tokens=config.cost_per_1k_tokens
        )
        
        try:
            embedding_provider = embedding_manager.get_provider(f"policy_{model_key}", provider_config)
            console.print(f"✓ Loaded model: {config.model_name}")
            
            # Create collection
            success = self.vector_store.create_collection(
                collection_name=config.name,
                dimension=config.dimension,
                distance_metric="cosine"
            )
            
            if success:
                console.print(f"✓ Created collection: {config.name}")
                console.print(f"✓ Model: {config.model_name} ({config.dimension}D)")
                console.print(f"✓ Description: {config.description}")
                return True
            else:
                console.print(f"[red]Failed to create collection: {config.name}[/red]")
                return False
                
        except Exception as e:
            console.print(f"[red]Error creating collection: {e}[/red]")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        return self._delete_collection(collection_name)
    
    def _delete_collection(self, collection_name: str) -> bool:
        """Internal method to delete collection."""
        try:
            response = requests.delete(f'http://localhost:6333/collections/{collection_name}', timeout=10)
            if response.status_code == 200:
                console.print(f"✓ Deleted collection: {collection_name}")
                return True
            else:
                console.print(f"[red]Failed to delete {collection_name}: {response.text}[/red]")
                return False
        except Exception as e:
            console.print(f"[red]Error deleting {collection_name}: {e}[/red]")
            return False
    
    def cleanup_old_collections(self):
        """Delete old collections that don't follow policyQA naming."""
        try:
            response = requests.get('http://localhost:6333/collections', timeout=5)
            data = response.json()
            collections = data['result']['collections']
            
            old_collections = [
                c['name'] for c in collections 
                if not c['name'].startswith('policyQA_')
            ]
            
            if not old_collections:
                console.print("[green]No old collections to clean up[/green]")
                return
            
            console.print(f"[yellow]Found {len(old_collections)} old collections:[/yellow]")
            for name in old_collections:
                console.print(f"  • {name}")
            
            if console.input("\nDelete these collections? [y/N]: ").lower().strip() == 'y':
                for name in old_collections:
                    self._delete_collection(name)
                console.print("[green]Cleanup complete![/green]")
            else:
                console.print("[dim]Cleanup cancelled[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error during cleanup: {e}[/red]")
    
    def get_model_for_collection(self, collection_name: str) -> Optional[ModelConfig]:
        """Get model config for a collection name."""
        for config in self.models.values():
            if config.name == collection_name:
                return config
        return None

def main():
    """Main CLI interface."""
    import sys
    
    manager = PolicyQACollectionManager()
    
    if len(sys.argv) < 2:
        console.print("[bold cyan]PolicyQA Collection Manager[/bold cyan]\n")
        console.print("Usage:")
        console.print("  python collection_manager.py list-models")
        console.print("  python collection_manager.py list-collections") 
        console.print("  python collection_manager.py create <model_key> [--force]")
        console.print("  python collection_manager.py delete <collection_name>")
        console.print("  python collection_manager.py cleanup")
        console.print("\nExample:")
        console.print("  python collection_manager.py create dense_qa768")
        return
    
    command = sys.argv[1]
    
    if command == "list-models":
        manager.list_available_models()
    
    elif command == "list-collections":
        manager.list_existing_collections()
    
    elif command == "create":
        if len(sys.argv) < 3:
            console.print("[red]Usage: create <model_key>[/red]")
            return
        
        model_key = sys.argv[2]
        force = "--force" in sys.argv
        manager.create_collection(model_key, force=force)
    
    elif command == "delete":
        if len(sys.argv) < 3:
            console.print("[red]Usage: delete <collection_name>[/red]")
            return
        
        collection_name = sys.argv[2]
        manager.delete_collection(collection_name)
    
    elif command == "cleanup":
        manager.cleanup_old_collections()
    
    else:
        console.print(f"[red]Unknown command: {command}[/red]")

if __name__ == "__main__":
    main()