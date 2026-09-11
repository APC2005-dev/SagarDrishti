"""environmental schema and feature-schema lineage

Additive only: nothing existing is dropped or rewritten except backfilling
``model_type = 'base'`` for the base row. Existing forecasts keep
``feature_schema_version = 'trajectory_v1'`` via the column default.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-11 22:36:30.283876
"""
from __future__ import annotations

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

IDENTITY_COLUMNS_0001 = [
    "version", "version_number", "parent_version", "architecture_version",
    "model_path", "scaler_path", "artifact_sha256", "training_data_cutoff",
]
IDENTITY_COLUMNS_0002 = IDENTITY_COLUMNS_0001 + [
    "model_type", "feature_schema_version", "feature_names", "environmental_data_sources", "environmental_data_cutoff",
]


def identity_fn(columns: list[str]) -> str:
    cond = "\n       OR ".join(f"NEW.{c} IS DISTINCT FROM OLD.{c}" for c in columns)
    return f"""
CREATE OR REPLACE FUNCTION ml.protect_model_identity() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF {cond} THEN
        RAISE EXCEPTION 'model version % is immutable (only status fields may change)', OLD.version;
    END IF;
    RETURN NEW;
END $$;
"""


ENV_APPEND_ONLY = [("observation_features", "UPDATE OR DELETE"), ("forecast_snapshots", "UPDATE OR DELETE")]


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS environmental")
    op.create_table('cache_entries',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('provider', sa.String(length=64), nullable=False),
    sa.Column('dataset_id', sa.String(length=128), nullable=False),
    sa.Column('group', sa.String(length=16), nullable=False),
    sa.Column('variables', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('tile_key', sa.String(length=16), nullable=False),
    sa.Column('period_start', sa.Date(), nullable=False),
    sa.Column('period_end', sa.Date(), nullable=False),
    sa.Column('bbox', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='POLYGON', srid=4326, dimension=2, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry', nullable=False), nullable=False),
    sa.Column('spatial_resolution_deg', sa.Float(), nullable=False),
    sa.Column('file_path', sa.Text(), nullable=False),
    sa.Column('sha256', sa.String(length=64), nullable=False),
    sa.Column('bytes', sa.BigInteger(), nullable=False),
    sa.Column('max_valid_date', sa.Date(), nullable=True),
    sa.Column('complete', sa.Boolean(), nullable=False),
    sa.Column('fetched_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_cache_entries')),
    sa.UniqueConstraint('provider', 'dataset_id', 'tile_key', 'period_start', name='uq_env_cache_entries_tile_period'),
    schema='environmental'
    )
    op.create_index('ix_env_cache_entries_geom', 'cache_entries', ['geom'], unique=False, schema='environmental', postgresql_using='gist')
    op.create_table('ingestion_runs',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('kind', sa.String(length=24), nullable=False),
    sa.Column('group', sa.String(length=16), nullable=True),
    sa.Column('provider', sa.String(length=64), nullable=True),
    sa.Column('dataset_id', sa.String(length=128), nullable=True),
    sa.Column('role', sa.String(length=16), nullable=True),
    sa.Column('request', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('records', sa.Integer(), server_default='0', nullable=False),
    sa.Column('bytes', sa.BigInteger(), nullable=True),
    sa.Column('cache_path', sa.Text(), nullable=True),
    sa.Column('duration_ms', sa.Integer(), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("kind IN ('tile_fetch', 'alignment', 'overlay', 'forecast_snapshot', 'backfill')", name=op.f('ck_ingestion_runs_kind')),
    sa.CheckConstraint("status IN ('running', 'success', 'partial', 'failed', 'skipped')", name=op.f('ck_ingestion_runs_status')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ingestion_runs')),
    schema='environmental'
    )
    op.create_index('ix_env_ingestion_runs_group_started', 'ingestion_runs', ['group', 'started_at'], unique=False, schema='environmental')
    op.create_table('observation_features',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('observation_id', sa.BigInteger(), nullable=False),
    sa.Column('iceberg_id', sa.String(length=32), nullable=False),
    sa.Column('group', sa.String(length=16), nullable=False),
    sa.Column('provider', sa.String(length=64), nullable=False),
    sa.Column('dataset_id', sa.String(length=128), nullable=False),
    sa.Column('values', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('units', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('observation_date', sa.Date(), nullable=False),
    sa.Column('as_of', sa.Date(), nullable=False),
    sa.Column('valid_date', sa.Date(), nullable=True),
    sa.Column('staleness_days', sa.Integer(), nullable=True),
    sa.Column('latitude', sa.Float(), nullable=False),
    sa.Column('longitude', sa.Float(), nullable=False),
    sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, dimension=2, spatial_index=False, from_text='ST_GeomFromEWKT', name='geometry', nullable=False), nullable=False),
    sa.Column('interpolation', sa.String(length=48), nullable=False),
    sa.Column('valid_neighbours', sa.Integer(), nullable=False),
    sa.Column('missing', sa.Boolean(), nullable=False),
    sa.Column('reason', sa.Text(), nullable=True),
    sa.Column('quality_flags', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('cache_path', sa.Text(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("\"group\" IN ('wind', 'current', 'sea_ice')", name=op.f('ck_observation_features_group')),
    sa.ForeignKeyConstraint(['observation_id'], ['tracking.observations.id'], name=op.f('fk_observation_features_observation_id_observations')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_observation_features')),
    sa.UniqueConstraint('observation_id', 'group', 'provider', 'as_of', name='uq_env_observation_features'),
    schema='environmental'
    )
    op.create_index('ix_env_observation_features_geom', 'observation_features', ['geom'], unique=False, schema='environmental', postgresql_using='gist')
    op.create_index(op.f('ix_environmental_observation_features_iceberg_id'), 'observation_features', ['iceberg_id'], unique=False, schema='environmental')
    op.create_index(op.f('ix_environmental_observation_features_observation_id'), 'observation_features', ['observation_id'], unique=False, schema='environmental')
    op.create_table('forecast_snapshots',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('forecast_set_id', sa.BigInteger(), nullable=False),
    sa.Column('group', sa.String(length=16), nullable=False),
    sa.Column('provider', sa.String(length=64), nullable=False),
    sa.Column('dataset_id', sa.String(length=128), nullable=False),
    sa.Column('issued_on', sa.Date(), nullable=False),
    sa.Column('lead_day', sa.Integer(), nullable=False),
    sa.Column('valid_date', sa.Date(), nullable=False),
    sa.Column('latitude', sa.Float(), nullable=False),
    sa.Column('longitude', sa.Float(), nullable=False),
    sa.Column('location_basis', sa.String(length=24), nullable=False),
    sa.Column('values', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
    sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['forecast_set_id'], ['ml.forecast_sets.id'], name=op.f('fk_forecast_snapshots_forecast_set_id_forecast_sets')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_forecast_snapshots')),
    sa.UniqueConstraint('forecast_set_id', 'group', 'lead_day', name='uq_env_forecast_snapshots'),
    schema='environmental'
    )
    op.create_index(op.f('ix_environmental_forecast_snapshots_forecast_set_id'), 'forecast_snapshots', ['forecast_set_id'], unique=False, schema='environmental')

    # --- additive columns on existing ml tables (defaults describe every existing row) ---
    op.add_column('forecast_sets', sa.Column('feature_schema_version', sa.String(length=48), server_default='trajectory_v1', nullable=False), schema='ml')
    op.add_column('forecast_sets', sa.Column('environment', postgresql.JSONB(astext_type=sa.Text()), nullable=True), schema='ml')
    op.add_column('forecast_sets', sa.Column('environment_as_of', sa.Date(), nullable=True), schema='ml')
    op.add_column('forecast_sets', sa.Column('fallback', postgresql.JSONB(astext_type=sa.Text()), nullable=True), schema='ml')
    op.add_column('model_versions', sa.Column('model_type', sa.String(length=16), server_default='trajectory', nullable=False), schema='ml')
    op.add_column('model_versions', sa.Column('feature_schema_version', sa.String(length=48), server_default='trajectory_v1', nullable=False), schema='ml')
    op.add_column('model_versions', sa.Column('environmental_data_sources', postgresql.JSONB(astext_type=sa.Text()), nullable=True), schema='ml')
    op.add_column('model_versions', sa.Column('environmental_data_cutoff', sa.Date(), nullable=True), schema='ml')
    op.create_index(op.f('ix_ml_model_versions_feature_schema_version'), 'model_versions', ['feature_schema_version'], unique=False, schema='ml')
    op.add_column('retraining_runs', sa.Column('experiment', postgresql.JSONB(astext_type=sa.Text()), nullable=True), schema='ml')

    # base keeps its identity; only its new classification column is set
    op.execute("UPDATE ml.model_versions SET model_type = 'base' WHERE version = 'base'")
    op.create_check_constraint(
        op.f('ck_model_versions_model_type'), 'model_versions', "model_type IN ('base', 'trajectory', 'environmental')", schema='ml'
    )
    op.execute(identity_fn(IDENTITY_COLUMNS_0002))

    op.execute("""
CREATE OR REPLACE FUNCTION environmental.forbid_modification() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% on %.% is not allowed: rows are permanent (append-only)', TG_OP, TG_TABLE_SCHEMA, TG_TABLE_NAME;
END $$;
""")
    for table, events in ENV_APPEND_ONLY:
        op.execute(
            f"CREATE TRIGGER {table}_forbid_modification BEFORE {events} ON environmental.{table} "
            f"FOR EACH ROW EXECUTE FUNCTION environmental.forbid_modification()"
        )


def downgrade() -> None:
    for table, _ in ENV_APPEND_ONLY:
        op.execute(f"DROP TRIGGER IF EXISTS {table}_forbid_modification ON environmental.{table}")
    op.execute("DROP FUNCTION IF EXISTS environmental.forbid_modification()")
    op.execute(identity_fn(IDENTITY_COLUMNS_0001))
    op.drop_constraint(op.f('ck_model_versions_model_type'), 'model_versions', schema='ml', type_='check')
    op.drop_column('retraining_runs', 'experiment', schema='ml')
    op.drop_index(op.f('ix_ml_model_versions_feature_schema_version'), table_name='model_versions', schema='ml')
    op.drop_column('model_versions', 'environmental_data_cutoff', schema='ml')
    op.drop_column('model_versions', 'environmental_data_sources', schema='ml')
    op.drop_column('model_versions', 'feature_schema_version', schema='ml')
    op.drop_column('model_versions', 'model_type', schema='ml')
    op.drop_column('forecast_sets', 'fallback', schema='ml')
    op.drop_column('forecast_sets', 'environment_as_of', schema='ml')
    op.drop_column('forecast_sets', 'environment', schema='ml')
    op.drop_column('forecast_sets', 'feature_schema_version', schema='ml')
    op.drop_index(op.f('ix_environmental_forecast_snapshots_forecast_set_id'), table_name='forecast_snapshots', schema='environmental')
    op.drop_table('forecast_snapshots', schema='environmental')
    op.drop_index(op.f('ix_environmental_observation_features_observation_id'), table_name='observation_features', schema='environmental')
    op.drop_index(op.f('ix_environmental_observation_features_iceberg_id'), table_name='observation_features', schema='environmental')
    op.drop_index('ix_env_observation_features_geom', table_name='observation_features', schema='environmental', postgresql_using='gist')
    op.drop_table('observation_features', schema='environmental')
    op.drop_index('ix_env_ingestion_runs_group_started', table_name='ingestion_runs', schema='environmental')
    op.drop_table('ingestion_runs', schema='environmental')
    op.drop_index('ix_env_cache_entries_geom', table_name='cache_entries', schema='environmental', postgresql_using='gist')
    op.drop_table('cache_entries', schema='environmental')
    op.execute("DROP SCHEMA IF EXISTS environmental")
