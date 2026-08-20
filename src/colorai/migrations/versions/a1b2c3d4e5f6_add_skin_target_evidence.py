"""add skin target evidence linkage and canonical profiles

Revision ID: a1b2c3d4e5f6
Revises: f5e6a7b8c9d0
Create Date: 2026-08-19 23:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'a1b2c3d4e5f6'
down_revision = 'f5e6a7b8c9d0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('skin_appearance_references', schema=None) as batch_op:
        batch_op.add_column(sa.Column('profile', sa.JSON(), nullable=True))

    with op.batch_alter_table('face_mask_tracks', schema=None) as batch_op:
        batch_op.add_column(sa.Column('human_approved', sa.Boolean(), nullable=False, server_default=sa.false()))

    with op.batch_alter_table('skin_appearance_targets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('mask_track_id', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('canonical_profile', sa.JSON(), nullable=False, server_default='{}'))
        batch_op.create_foreign_key(
            batch_op.f('fk_skin_appearance_targets_mask_track_id_face_mask_tracks'),
            'face_mask_tracks', ['mask_track_id'], ['id'], ondelete='SET NULL'
        )


def downgrade() -> None:
    with op.batch_alter_table('skin_appearance_targets', schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f('fk_skin_appearance_targets_mask_track_id_face_mask_tracks'),
            type_='foreignkey'
        )
        batch_op.drop_column('canonical_profile')
        batch_op.drop_column('mask_track_id')

    with op.batch_alter_table('face_mask_tracks', schema=None) as batch_op:
        batch_op.drop_column('human_approved')

    with op.batch_alter_table('skin_appearance_references', schema=None) as batch_op:
        batch_op.drop_column('profile')
