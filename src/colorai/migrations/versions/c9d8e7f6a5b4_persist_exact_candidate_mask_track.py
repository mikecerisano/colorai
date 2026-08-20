"""persist exact candidate mask track on face corrections

Revision ID: c9d8e7f6a5b4
Revises: a1b2c3d4e5f6
Create Date: 2026-08-20 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'c9d8e7f6a5b4'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('face_corrections', schema=None) as batch_op:
        batch_op.add_column(sa.Column('mask_track_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            batch_op.f('fk_face_corrections_mask_track_id_face_mask_tracks'),
            'face_mask_tracks', ['mask_track_id'], ['id'], ondelete='SET NULL'
        )


def downgrade() -> None:
    with op.batch_alter_table('face_corrections', schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f('fk_face_corrections_mask_track_id_face_mask_tracks'),
            type_='foreignkey'
        )
        batch_op.drop_column('mask_track_id')
