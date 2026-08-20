"""add skin targets and mask tracks

Revision ID: f5e6a7b8c9d0
Revises: d3e4f5a6b7c8
Create Date: 2026-08-19 23:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'f5e6a7b8c9d0'
down_revision = 'd3e4f5a6b7c8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('skin_appearance_references',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=True),
        sa.Column('group_id', sa.Integer(), nullable=True),
        sa.Column('source_kind', sa.String(length=24), nullable=False),
        sa.Column('role', sa.String(length=32), nullable=False),
        sa.Column('source_path', sa.Text(), nullable=True),
        sa.Column('managed_path', sa.Text(), nullable=True),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('source_shot_id', sa.Integer(), nullable=True),
        sa.Column('frame_index', sa.Integer(), nullable=True),
        sa.Column('crop_geometry', sa.JSON(), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['asset_id'], ['media_assets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['group_id'], ['shot_groups.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['source_shot_id'], ['shots.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('skin_appearance_references', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_skin_appearance_references_subject_id'), ['subject_id'], unique=False)

    op.create_table('face_mask_tracks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('face_track_id', sa.Integer(), nullable=False),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=False),
        sa.Column('backend', sa.String(length=32), nullable=False),
        sa.Column('backend_version', sa.String(length=64), nullable=False),
        sa.Column('strategy', sa.String(length=32), nullable=False),
        sa.Column('landmark_keyframes', sa.JSON(), nullable=False),
        sa.Column('coverage', sa.Float(), nullable=False),
        sa.Column('max_gap', sa.Float(), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('review_state', sa.String(length=32), nullable=False),
        sa.Column('review_reason', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['face_track_id'], ['face_tracks.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('face_track_id')
    )
    with op.batch_alter_table('face_mask_tracks', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_face_mask_tracks_shot_id'), ['shot_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_face_mask_tracks_subject_id'), ['subject_id'], unique=False)

    op.create_table('skin_appearance_targets',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=False),
        sa.Column('group_id', sa.Integer(), nullable=False),
        sa.Column('reference_id', sa.Integer(), nullable=True),
        sa.Column('profile', sa.JSON(), nullable=False),
        sa.Column('approved_preview_parameters', sa.JSON(), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('rationale', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['group_id'], ['shot_groups.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reference_id'], ['skin_appearance_references.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('skin_appearance_targets', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_skin_appearance_targets_group_id'), ['group_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_skin_appearance_targets_subject_id'), ['subject_id'], unique=False)

    with op.batch_alter_table('face_corrections', schema=None) as batch_op:
        batch_op.add_column(sa.Column('skin_target_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            batch_op.f('fk_face_corrections_skin_target_id_skin_appearance_targets'),
            'skin_appearance_targets', ['skin_target_id'], ['id'], ondelete='SET NULL'
        )


def downgrade() -> None:
    with op.batch_alter_table('face_corrections', schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f('fk_face_corrections_skin_target_id_skin_appearance_targets'),
            type_='foreignkey'
        )
        batch_op.drop_column('skin_target_id')

    with op.batch_alter_table('skin_appearance_targets', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_skin_appearance_targets_subject_id'))
        batch_op.drop_index(batch_op.f('ix_skin_appearance_targets_group_id'))
    op.drop_table('skin_appearance_targets')

    with op.batch_alter_table('face_mask_tracks', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_face_mask_tracks_subject_id'))
        batch_op.drop_index(batch_op.f('ix_face_mask_tracks_shot_id'))
    op.drop_table('face_mask_tracks')

    with op.batch_alter_table('skin_appearance_references', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_skin_appearance_references_subject_id'))
    op.drop_table('skin_appearance_references')
