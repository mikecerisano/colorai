"""persist face bounding boxes on skin metrics

Revision ID: f3c5d7e9a1b2
Revises: e7a2b4c5d6f8
Create Date: 2026-08-19 19:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'f3c5d7e9a1b2'
down_revision = 'e7a2b4c5d6f8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('skin_metrics', schema=None) as batch_op:
        batch_op.add_column(sa.Column('bbox_x', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('bbox_y', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('bbox_w', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('bbox_h', sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('skin_metrics', schema=None) as batch_op:
        batch_op.drop_column('bbox_h')
        batch_op.drop_column('bbox_w')
        batch_op.drop_column('bbox_y')
        batch_op.drop_column('bbox_x')
