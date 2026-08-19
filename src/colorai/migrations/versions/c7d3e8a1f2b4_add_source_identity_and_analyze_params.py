"""add source identity and analyze params

Revision ID: c7d3e8a1f2b4
Revises: 0026a3722cec
Create Date: 2026-08-19 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'c7d3e8a1f2b4'
down_revision = '0026a3722cec'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('media_assets', schema=None) as batch_op:
        batch_op.add_column(sa.Column('source_hash', sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column('analyze_params', sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('media_assets', schema=None) as batch_op:
        batch_op.drop_column('analyze_params')
        batch_op.drop_column('source_hash')
