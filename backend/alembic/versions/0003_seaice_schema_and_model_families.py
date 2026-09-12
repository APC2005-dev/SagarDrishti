"""sea-ice schema and per-family model version lineages

Two changes, both additive in effect:

1. A new ``seaice`` schema holding official Copernicus Marine sea-ice grids,
   the forecasts produced from them and their evaluations. Observations are
   append-only.
2. ``ml.model_versions`` gains ``model_family`` so the trajectory and sea-ice
   models keep INDEPENDENT lineages. Version numbers become unique per family
   instead of globally, and "one champion" becomes "one champion per family".
   ``version`` stays globally unique (sea-ice rows are stored qualified, e.g.
   ``sea_ice/v1``) so every existing foreign key to it is untouched.

Existing rows are backfilled to ``model_family = 'trajectory'``, which is what
they are; no trajectory behaviour changes.

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-13
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

IDENTITY_COLUMNS_0002 = [
    "version", "version_number", "parent_version", "architecture_version",
    "model_path", "scaler_path", "artifact_sha256", "training_data_cutoff",
    "model_type", "feature_schema_version", "feature_names", "environmental_data_sources",
    "environmental_data_cutoff",
]
# A version may never change the family it belongs to.
IDENTITY_COLUMNS_0003 = IDENTITY_COLUMNS_0002 + ["model_family"]


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


APPEND_ONLY = """
CREATE OR REPLACE FUNCTION seaice.append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'seaice.% is append-only: official observations are never updated or deleted', TG_TABLE_NAME;
END $$;
"""


def upgrade() -> None:
    # ---------------------------------------------------------------- seaice
    op.execute("CREATE SCHEMA IF NOT EXISTS seaice")
    op.create_table(
        'runs',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('trigger', sa.String(length=32), server_default='scheduled', nullable=False),
        sa.Column('dataset_id', sa.String(length=128), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('duration_ms', sa.Integer(), nullable=True),
        sa.Column('source_latest_date', sa.Date(), nullable=True),
        sa.Column('entries_seen', sa.Integer(), server_default='0', nullable=False),
        sa.Column('entries_new', sa.Integer(), server_default='0', nullable=False),
        sa.Column('entries_duplicate', sa.Integer(), server_default='0', nullable=False),
        sa.Column('status', sa.String(length=16), server_default='running', nullable=False),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint("kind IN ('ingestion', 'forecast', 'evaluation', 'retraining')", name=op.f('ck_runs_kind')),
        sa.CheckConstraint(
            "status IN ('running', 'success', 'unchanged', 'partial', 'failed', 'skipped')", name=op.f('ck_runs_status')
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_runs')),
        schema='seaice',
    )
    op.create_index('ix_seaice_runs_kind_started', 'runs', ['kind', 'started_at'], schema='seaice')

    op.create_table(
        'observations',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('observation_date', sa.Date(), nullable=False),
        sa.Column('source_time', sa.DateTime(timezone=True), nullable=True),
        sa.Column('provenance', sa.String(length=32), server_default='official_cmems_osisaf', nullable=False),
        sa.Column('authority', sa.String(length=128), nullable=False),
        sa.Column('dataset_id', sa.String(length=128), nullable=False),
        sa.Column('variable', sa.String(length=32), nullable=False),
        sa.Column('preprocessing_version', sa.String(length=32), nullable=False),
        sa.Column('grid_shape', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('grid_resolution_deg', sa.Float(), nullable=False),
        sa.Column('crs', sa.String(length=32), nullable=False),
        sa.Column('grid_path', sa.Text(), nullable=False),
        sa.Column('grid_sha256', sa.String(length=64), nullable=False),
        sa.Column('n_valid_cells', sa.Integer(), nullable=False),
        sa.Column('mean_concentration', sa.Float(), nullable=True),
        sa.Column('ice_area_km2', sa.Float(), nullable=True),
        sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('ingestion_run_id', sa.BigInteger(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint("provenance IN ('official_cmems_osisaf')", name=op.f('ck_observations_provenance')),
        sa.CheckConstraint('n_valid_cells >= 0', name=op.f('ck_observations_n_valid_cells')),
        sa.ForeignKeyConstraint(
            ['ingestion_run_id'], ['seaice.runs.id'], name=op.f('fk_observations_ingestion_run_id_runs')
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_observations')),
        sa.UniqueConstraint('observation_date', name='uq_seaice_observations_date'),
        schema='seaice',
    )
    op.create_index('ix_seaice_observations_date', 'observations', ['observation_date'], schema='seaice')

    op.create_table(
        'forecast_sets',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('model_version', sa.String(length=32), nullable=False),
        sa.Column('anchor_observation_id', sa.BigInteger(), nullable=False),
        sa.Column('anchor_date', sa.Date(), nullable=False),
        sa.Column('input_observation_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('input_entry_dates', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('input_window_entries', sa.Integer(), nullable=False),
        sa.Column('input_span_days', sa.Integer(), nullable=True),
        sa.Column('daily_cadence', sa.Boolean(), nullable=True),
        sa.Column('generated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('run_id', sa.BigInteger(), nullable=True),
        sa.Column('diagnostics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(
            ['anchor_observation_id'], ['seaice.observations.id'],
            name=op.f('fk_forecast_sets_anchor_observation_id_observations'),
        ),
        sa.ForeignKeyConstraint(
            ['model_version'], ['ml.model_versions.version'], name=op.f('fk_forecast_sets_model_version_model_versions')
        ),
        sa.ForeignKeyConstraint(['run_id'], ['seaice.runs.id'], name=op.f('fk_forecast_sets_run_id_runs')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_forecast_sets')),
        sa.UniqueConstraint('model_version', 'anchor_observation_id', name='uq_seaice_forecast_sets_model_anchor'),
        schema='seaice',
    )
    op.create_index(op.f('ix_seaice_forecast_sets_anchor_date'), 'forecast_sets', ['anchor_date'], schema='seaice')
    op.create_index(op.f('ix_seaice_forecast_sets_model_version'), 'forecast_sets', ['model_version'], schema='seaice')

    op.create_table(
        'forecasts',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('forecast_set_id', sa.BigInteger(), nullable=False),
        sa.Column('horizon_days', sa.Integer(), nullable=False),
        sa.Column('target_date', sa.Date(), nullable=False),
        sa.Column('grid_path', sa.Text(), nullable=False),
        sa.Column('grid_sha256', sa.String(length=64), nullable=False),
        sa.Column('mean_concentration', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.CheckConstraint('horizon_days > 0', name=op.f('ck_forecasts_horizon_days')),
        sa.ForeignKeyConstraint(
            ['forecast_set_id'], ['seaice.forecast_sets.id'], name=op.f('fk_forecasts_forecast_set_id_forecast_sets')
        ),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_forecasts')),
        sa.UniqueConstraint('forecast_set_id', 'horizon_days', name='uq_seaice_forecasts_set_horizon'),
        schema='seaice',
    )
    op.create_index(op.f('ix_seaice_forecasts_forecast_set_id'), 'forecasts', ['forecast_set_id'], schema='seaice')
    op.create_index(op.f('ix_seaice_forecasts_target_date'), 'forecasts', ['target_date'], schema='seaice')

    op.create_table(
        'evaluations',
        sa.Column('id', sa.BigInteger(), nullable=False),
        sa.Column('forecast_id', sa.BigInteger(), nullable=False),
        sa.Column('model_version', sa.String(length=32), nullable=False),
        sa.Column('horizon_days', sa.Integer(), nullable=False),
        sa.Column('anchor_date', sa.Date(), nullable=False),
        sa.Column('target_date', sa.Date(), nullable=False),
        sa.Column('actual_observation_id', sa.BigInteger(), nullable=False),
        sa.Column('rmse', sa.Float(), nullable=False),
        sa.Column('mae', sa.Float(), nullable=False),
        sa.Column('persistence_rmse', sa.Float(), nullable=True),
        sa.Column('persistence_mae', sa.Float(), nullable=True),
        sa.Column('n_valid_cells', sa.Integer(), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('run_id', sa.BigInteger(), nullable=True),
        sa.ForeignKeyConstraint(
            ['actual_observation_id'], ['seaice.observations.id'],
            name=op.f('fk_evaluations_actual_observation_id_observations'),
        ),
        sa.ForeignKeyConstraint(
            ['forecast_id'], ['seaice.forecasts.id'], name=op.f('fk_evaluations_forecast_id_forecasts')
        ),
        sa.ForeignKeyConstraint(
            ['model_version'], ['ml.model_versions.version'], name=op.f('fk_evaluations_model_version_model_versions')
        ),
        sa.ForeignKeyConstraint(['run_id'], ['seaice.runs.id'], name=op.f('fk_evaluations_run_id_runs')),
        sa.PrimaryKeyConstraint('id', name=op.f('pk_evaluations')),
        sa.UniqueConstraint('forecast_id', 'actual_observation_id', name='uq_seaice_evaluations_forecast_actual'),
        schema='seaice',
    )
    op.create_index('ix_seaice_evaluations_model_horizon', 'evaluations', ['model_version', 'horizon_days'], schema='seaice')
    op.create_index(op.f('ix_seaice_evaluations_evaluated_at'), 'evaluations', ['evaluated_at'], schema='seaice')
    op.create_index(op.f('ix_seaice_evaluations_forecast_id'), 'evaluations', ['forecast_id'], schema='seaice')

    # Official sea-ice observations are never rewritten.
    op.execute(APPEND_ONLY)
    op.execute(
        "CREATE TRIGGER observations_append_only BEFORE UPDATE OR DELETE ON seaice.observations "
        "FOR EACH ROW EXECUTE FUNCTION seaice.append_only()"
    )

    # -------------------------------------------------- per-family lineages
    op.add_column(
        'model_versions',
        sa.Column('model_family', sa.String(length=16), server_default='trajectory', nullable=False),
        schema='ml',
    )
    op.execute("UPDATE ml.model_versions SET model_family = 'trajectory'")
    op.create_index(op.f('ix_ml_model_versions_model_family'), 'model_versions', ['model_family'], schema='ml')
    op.create_check_constraint(
        'model_family', 'model_versions', "model_family IN ('trajectory', 'sea_ice')", schema='ml'
    )
    op.drop_constraint(op.f('ck_model_versions_model_type'), 'model_versions', schema='ml', type_='check')
    op.create_check_constraint(
        'model_type', 'model_versions',
        "model_type IN ('base', 'trajectory', 'environmental', 'sea_ice')", schema='ml',
    )
    # Version numbers restart at 1 in every family.
    op.drop_constraint(op.f('uq_model_versions_version_number'), 'model_versions', schema='ml', type_='unique')
    op.create_unique_constraint(
        op.f('uq_model_versions_family_version_number'), 'model_versions', ['model_family', 'version_number'], schema='ml'
    )
    # One champion PER FAMILY, instead of one globally.
    op.drop_index(
        'uq_model_versions_single_deployed', table_name='model_versions', schema='ml',
        postgresql_where=sa.text("status = 'deployed'"),
    )
    op.create_index(
        'uq_model_versions_single_deployed', 'model_versions', ['model_family'], unique=True, schema='ml',
        postgresql_where=sa.text("status = 'deployed'"),
    )
    op.execute(identity_fn(IDENTITY_COLUMNS_0003))


def downgrade() -> None:
    op.execute(identity_fn(IDENTITY_COLUMNS_0002))
    op.drop_index(
        'uq_model_versions_single_deployed', table_name='model_versions', schema='ml',
        postgresql_where=sa.text("status = 'deployed'"),
    )
    op.create_index(
        'uq_model_versions_single_deployed', 'model_versions', ['status'], unique=True, schema='ml',
        postgresql_where=sa.text("status = 'deployed'"),
    )
    op.drop_constraint(op.f('uq_model_versions_family_version_number'), 'model_versions', schema='ml', type_='unique')
    op.create_unique_constraint(
        op.f('uq_model_versions_version_number'), 'model_versions', ['version_number'], schema='ml'
    )
    op.drop_constraint(op.f('ck_model_versions_model_type'), 'model_versions', schema='ml', type_='check')
    op.create_check_constraint(
        'model_type', 'model_versions', "model_type IN ('base', 'trajectory', 'environmental')", schema='ml'
    )
    op.drop_constraint(op.f('ck_model_versions_model_family'), 'model_versions', schema='ml', type_='check')
    op.drop_index(op.f('ix_ml_model_versions_model_family'), table_name='model_versions', schema='ml')
    op.drop_column('model_versions', 'model_family', schema='ml')

    op.execute("DROP TRIGGER IF EXISTS observations_append_only ON seaice.observations")
    op.drop_table('evaluations', schema='seaice')
    op.drop_table('forecasts', schema='seaice')
    op.drop_table('forecast_sets', schema='seaice')
    op.drop_table('observations', schema='seaice')
    op.drop_table('runs', schema='seaice')
    op.execute("DROP FUNCTION IF EXISTS seaice.append_only()")
    op.execute("DROP SCHEMA IF EXISTS seaice CASCADE")
