"""add face tracks and local face corrections

Revision ID: d3e4f5a6b7c8
Revises: c1d2e3f4a5b6
Create Date: 2026-08-19 22:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'd3e4f5a6b7c8'
down_revision = 'c1d2e3f4a5b6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('face_tracks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('skin_metric_id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=True),
        sa.Column('source_width', sa.Integer(), nullable=False),
        sa.Column('source_height', sa.Integer(), nullable=False),
        sa.Column('analysis_scale', sa.Integer(), nullable=True),
        sa.Column('keyframes', sa.JSON(), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=False),
        sa.Column('tracked_count', sa.Integer(), nullable=False),
        sa.Column('coverage', sa.Float(), nullable=False),
        sa.Column('max_gap', sa.Float(), nullable=False),
        sa.Column('skin_stability', sa.Float(), nullable=False),
        sa.Column('median_bgr', sa.JSON(), nullable=True),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['skin_metric_id'], ['skin_metrics.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('face_tracks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_face_tracks_shot_id'), ['shot_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_face_tracks_skin_metric_id'), ['skin_metric_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_face_tracks_subject_id'), ['subject_id'], unique=False)

    op.create_table('face_corrections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=True),
        sa.Column('skin_metric_id', sa.Integer(), nullable=True),
        sa.Column('face_track_id', sa.Integer(), nullable=True),
        sa.Column('reference_shot_id', sa.Integer(), nullable=True),
        sa.Column('reference_group_id', sa.Integer(), nullable=True),
        sa.Column('kind', sa.String(length=64), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('evidence', sa.JSON(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('classification', sa.String(length=32), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['face_track_id'], ['face_tracks.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reference_group_id'], ['shot_groups.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['reference_shot_id'], ['shots.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['skin_metric_id'], ['skin_metrics.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('face_corrections', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_face_corrections_shot_id'), ['shot_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_face_corrections_subject_id'), ['subject_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('face_corrections', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_face_corrections_subject_id'))
        batch_op.drop_index(batch_op.f('ix_face_corrections_shot_id'))

    op.drop_table('face_corrections')

    with op.batch_alter_table('face_tracks', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_face_tracks_subject_id'))
        batch_op.drop_index(batch_op.f('ix_face_tracks_skin_metric_id'))
        batch_op.drop_index(batch_op.f('ix_face_tracks_shot_id'))

    op.drop_table('face_tracks')
