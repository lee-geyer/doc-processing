"""Initial database schema

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create file_sync_state table
    op.create_table('file_sync_state',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('file_path', sa.String(length=500), nullable=False),
    sa.Column('file_hash', sa.String(length=64), nullable=False),
    sa.Column('file_size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('last_modified', sa.DateTime(), nullable=False),
    sa.Column('sync_status', sa.String(length=20), nullable=False),
    sa.Column('last_sync_at', sa.DateTime(), nullable=True),
    sa.Column('sync_error_count', sa.Integer(), nullable=True),
    sa.Column('last_error_message', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('file_path')
    )
    op.create_index('idx_file_sync_state_modified', 'file_sync_state', ['last_modified'], unique=False)
    op.create_index('idx_file_sync_state_path', 'file_sync_state', ['file_path'], unique=False)
    op.create_index('idx_file_sync_state_status', 'file_sync_state', ['sync_status'], unique=False)

    # Create embedding_models table
    op.create_table('embedding_models',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('provider', sa.String(length=50), nullable=False),
    sa.Column('model_name', sa.String(length=100), nullable=False),
    sa.Column('dimension', sa.Integer(), nullable=False),
    sa.Column('max_tokens', sa.Integer(), nullable=True),
    sa.Column('cost_per_1k_tokens', sa.Float(), nullable=True),
    sa.Column('is_active', sa.Boolean(), nullable=True),
    sa.Column('avg_retrieval_accuracy', sa.Float(), nullable=True),
    sa.Column('avg_processing_speed_ms', sa.Float(), nullable=True),
    sa.Column('total_cost_usd', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('provider', 'model_name')
    )

    # Create vector_collections table
    op.create_table('vector_collections',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('collection_name', sa.String(length=100), nullable=False),
    sa.Column('embedding_model_id', sa.Integer(), nullable=False),
    sa.Column('dimension', sa.Integer(), nullable=False),
    sa.Column('total_vectors', sa.Integer(), nullable=True),
    sa.Column('last_sync_at', sa.DateTime(), nullable=True),
    sa.Column('distance_metric', sa.String(length=20), nullable=True),
    sa.Column('index_config', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['embedding_model_id'], ['embedding_models.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('collection_name')
    )

    # Create documents table
    op.create_table('documents',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('file_path', sa.String(length=500), nullable=False),
    sa.Column('document_index', sa.String(length=100), nullable=True),
    sa.Column('file_hash', sa.String(length=64), nullable=False),
    sa.Column('original_filename', sa.String(length=255), nullable=False),
    sa.Column('file_type', sa.String(length=50), nullable=False),
    sa.Column('file_size_bytes', sa.BigInteger(), nullable=False),
    sa.Column('parsing_method', sa.String(length=50), nullable=False),
    sa.Column('parsing_success', sa.Boolean(), nullable=False),
    sa.Column('parsing_error_message', sa.Text(), nullable=True),
    sa.Column('markdown_content', sa.Text(), nullable=True),
    sa.Column('cleaned_content', sa.Text(), nullable=True),
    sa.Column('word_count', sa.Integer(), nullable=True),
    sa.Column('page_count', sa.Integer(), nullable=True),
    sa.Column('document_metadata', sa.JSON(), nullable=True),
    sa.Column('context_hierarchy', sa.JSON(), nullable=True),
    sa.Column('total_chunks', sa.Integer(), nullable=True),
    sa.Column('total_tokens', sa.Integer(), nullable=True),
    sa.Column('processing_duration_ms', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('file_path')
    )
    op.create_index('idx_documents_hash', 'documents', ['file_hash'], unique=False)
    op.create_index('idx_documents_index', 'documents', ['document_index'], unique=False)
    op.create_index('idx_documents_path', 'documents', ['file_path'], unique=False)

    # Create document_chunks table
    op.create_table('document_chunks',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('chunk_text', sa.Text(), nullable=False),
    sa.Column('chunk_markdown', sa.Text(), nullable=True),
    sa.Column('chunk_tokens', sa.Integer(), nullable=False),
    sa.Column('chunk_hash', sa.String(length=64), nullable=False),
    sa.Column('page_numbers', postgresql.ARRAY(sa.Integer()), nullable=True),
    sa.Column('section_title', sa.String(length=255), nullable=True),
    sa.Column('subsection_title', sa.String(length=255), nullable=True),
    sa.Column('context_metadata', sa.JSON(), nullable=False),
    sa.Column('surrounding_context', sa.Text(), nullable=True),
    sa.Column('document_position', sa.Float(), nullable=True),
    sa.Column('vector_id', postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column('embedding_model_id', sa.Integer(), nullable=True),
    sa.Column('vector_collection_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['embedding_model_id'], ['embedding_models.id'], ),
    sa.ForeignKeyConstraint(['vector_collection_id'], ['vector_collections.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('document_id', 'chunk_index')
    )
    op.create_index('idx_chunks_document', 'document_chunks', ['document_id'], unique=False)
    op.create_index('idx_chunks_hash', 'document_chunks', ['chunk_hash'], unique=False)
    op.create_index('idx_chunks_vector', 'document_chunks', ['vector_id'], unique=False)

    # Create vector_sync_operations table
    op.create_table('vector_sync_operations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('operation_type', sa.String(length=20), nullable=False),
    sa.Column('file_path', sa.String(length=500), nullable=True),
    sa.Column('document_id', sa.Integer(), nullable=True),
    sa.Column('chunks_affected', sa.Integer(), nullable=True),
    sa.Column('vector_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
    sa.Column('collection_id', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('execution_time_ms', sa.Integer(), nullable=True),
    sa.Column('retry_count', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['collection_id'], ['vector_collections.id'], ),
    sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_sync_ops_created', 'vector_sync_operations', ['created_at'], unique=False)
    op.create_index('idx_sync_ops_status', 'vector_sync_operations', ['status'], unique=False)

    # Create embedding_evaluations table
    op.create_table('embedding_evaluations',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('evaluation_name', sa.String(length=100), nullable=False),
    sa.Column('embedding_model_id', sa.Integer(), nullable=False),
    sa.Column('test_query_count', sa.Integer(), nullable=False),
    sa.Column('evaluation_date', sa.DateTime(), nullable=False),
    sa.Column('precision_at_5', sa.Float(), nullable=True),
    sa.Column('recall_at_10', sa.Float(), nullable=True),
    sa.Column('avg_response_time_ms', sa.Float(), nullable=True),
    sa.Column('semantic_clustering_score', sa.Float(), nullable=True),
    sa.Column('cross_document_score', sa.Float(), nullable=True),
    sa.Column('total_cost_usd', sa.Float(), nullable=True),
    sa.Column('cost_per_query', sa.Float(), nullable=True),
    sa.Column('detailed_results', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['embedding_model_id'], ['embedding_models.id'], ),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('embedding_evaluations')
    op.drop_index('idx_sync_ops_status', table_name='vector_sync_operations')
    op.drop_index('idx_sync_ops_created', table_name='vector_sync_operations')
    op.drop_table('vector_sync_operations')
    op.drop_index('idx_chunks_vector', table_name='document_chunks')
    op.drop_index('idx_chunks_hash', table_name='document_chunks')
    op.drop_index('idx_chunks_document', table_name='document_chunks')
    op.drop_table('document_chunks')
    op.drop_index('idx_documents_path', table_name='documents')
    op.drop_index('idx_documents_index', table_name='documents')
    op.drop_index('idx_documents_hash', table_name='documents')
    op.drop_table('documents')
    op.drop_table('vector_collections')
    op.drop_table('embedding_models')
    op.drop_index('idx_file_sync_state_status', table_name='file_sync_state')
    op.drop_index('idx_file_sync_state_path', table_name='file_sync_state')
    op.drop_index('idx_file_sync_state_modified', table_name='file_sync_state')
    op.drop_table('file_sync_state')