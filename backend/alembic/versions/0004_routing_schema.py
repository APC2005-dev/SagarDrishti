"""routing schema: canonical ports and computed routes

Additive: a new ``routing`` schema only. No existing table is touched.
Ports cache the NGA World Port Index; routes persist one A* computation each
with the model versions and forecast runs that produced it.

Revision ID: c905b3dae8b2
Revises: 0003
Create Date: 2026-09-13 06:00:36.350238
"""
from __future__ import annotations

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE SCHEMA IF NOT EXISTS routing")
    op.create_table('ports',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('source', sa.String(length=32), server_default='nga_world_port_index', nullable=False),
    sa.Column('identifier', sa.String(length=32), nullable=False),
    sa.Column('name', sa.String(length=128), nullable=False),
    sa.Column('normalized_name', sa.String(length=128), nullable=False),
    sa.Column('unlocode', sa.String(length=16), nullable=True),
    sa.Column('country_code', sa.String(length=8), nullable=True),
    sa.Column('country_name', sa.String(length=96), nullable=True),
    sa.Column('region_name', sa.String(length=96), nullable=True),
    sa.Column('latitude', sa.Float(), nullable=False),
    sa.Column('longitude', sa.Float(), nullable=False),
    sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='POINT', srid=4326, dimension=2, from_text='ST_GeomFromEWKT', name='geometry', nullable=False), nullable=False),
    sa.Column('attributes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('refreshed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("source IN ('nga_world_port_index')", name=op.f('ck_ports_source')),
    sa.CheckConstraint('latitude BETWEEN -90 AND 90', name=op.f('ck_ports_latitude')),
    sa.CheckConstraint('longitude BETWEEN -180 AND 180', name=op.f('ck_ports_longitude')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_ports')),
    sa.UniqueConstraint('source', 'identifier', name='uq_routing_ports_source_identifier'),
    schema='routing'
    )
    op.create_index('ix_routing_ports_normalized_name', 'ports', ['normalized_name'], unique=False, schema='routing')
    op.create_table('routes',
    sa.Column('id', sa.BigInteger(), nullable=False),
    sa.Column('route_id', sa.String(length=36), nullable=False),
    sa.Column('status', sa.String(length=24), server_default='VALIDATING', nullable=False),
    sa.Column('error_code', sa.String(length=48), nullable=True),
    sa.Column('error_message', sa.Text(), nullable=True),
    sa.Column('departure_port_id', sa.BigInteger(), nullable=False),
    sa.Column('destination_port_id', sa.BigInteger(), nullable=False),
    sa.Column('requested_departure_latitude', sa.Float(), nullable=False),
    sa.Column('requested_departure_longitude', sa.Float(), nullable=False),
    sa.Column('requested_destination_latitude', sa.Float(), nullable=False),
    sa.Column('requested_destination_longitude', sa.Float(), nullable=False),
    sa.Column('resolved_departure_latitude', sa.Float(), nullable=True),
    sa.Column('resolved_departure_longitude', sa.Float(), nullable=True),
    sa.Column('resolved_destination_latitude', sa.Float(), nullable=True),
    sa.Column('resolved_destination_longitude', sa.Float(), nullable=True),
    sa.Column('departure_connector_km', sa.Float(), nullable=True),
    sa.Column('destination_connector_km', sa.Float(), nullable=True),
    sa.Column('connectors_validated', sa.Boolean(), nullable=True),
    sa.Column('trajectory_model_version', sa.String(length=32), nullable=True),
    sa.Column('trajectory_forecast_run_id', sa.BigInteger(), nullable=True),
    sa.Column('sea_ice_model_version', sa.String(length=32), nullable=True),
    sa.Column('sea_ice_forecast_run_id', sa.BigInteger(), nullable=True),
    sa.Column('route_planner_version', sa.String(length=32), nullable=False),
    sa.Column('forecast_reference_time', sa.DateTime(timezone=True), nullable=True),
    sa.Column('environment_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('distance_km', sa.Float(), nullable=True),
    sa.Column('duration_hours', sa.Float(), nullable=True),
    sa.Column('fuel_proxy', sa.Float(), nullable=True),
    sa.Column('sic_exposure_hours', sa.Float(), nullable=True),
    sa.Column('weighted_objective', sa.Float(), nullable=True),
    sa.Column('expansions', sa.Integer(), nullable=True),
    sa.Column('runtime_seconds', sa.Float(), nullable=True),
    sa.Column('confidence_status', sa.String(length=24), server_default='not_calibrated', nullable=False),
    sa.Column('curvature', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('geom', geoalchemy2.types.Geometry(geometry_type='LINESTRING', srid=4326, dimension=2, from_text='ST_GeomFromEWKT', name='geometry'), nullable=True),
    sa.Column('waypoints', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.CheckConstraint("status IN ('VALIDATING', 'RUNNING', 'ROUTE_FOUND', 'NO_FEASIBLE_ROUTE', 'FAILED', 'ACTIVE', 'COMPLETED')", name=op.f('ck_routes_status')),
    sa.CheckConstraint('departure_port_id <> destination_port_id', name=op.f('ck_routes_distinct_ports')),
    sa.ForeignKeyConstraint(['departure_port_id'], ['routing.ports.id'], name=op.f('fk_routes_departure_port_id_ports')),
    sa.ForeignKeyConstraint(['destination_port_id'], ['routing.ports.id'], name=op.f('fk_routes_destination_port_id_ports')),
    sa.ForeignKeyConstraint(['sea_ice_model_version'], ['ml.model_versions.version'], name=op.f('fk_routes_sea_ice_model_version_model_versions')),
    sa.ForeignKeyConstraint(['trajectory_model_version'], ['ml.model_versions.version'], name=op.f('fk_routes_trajectory_model_version_model_versions')),
    sa.PrimaryKeyConstraint('id', name=op.f('pk_routes')),
    schema='routing'
    )
    op.create_index(op.f('ix_routing_routes_route_id'), 'routes', ['route_id'], unique=True, schema='routing')
    op.create_index('ix_routing_routes_status_created', 'routes', ['status', 'created_at'], unique=False, schema='routing')


def downgrade() -> None:
    op.drop_index('ix_routing_routes_status_created', table_name='routes', schema='routing')
    op.drop_index(op.f('ix_routing_routes_route_id'), table_name='routes', schema='routing')
    op.drop_table('routes', schema='routing')
    op.drop_index('ix_routing_ports_normalized_name', table_name='ports', schema='routing')
    op.drop_table('ports', schema='routing')
    op.execute("DROP SCHEMA IF EXISTS routing CASCADE")
