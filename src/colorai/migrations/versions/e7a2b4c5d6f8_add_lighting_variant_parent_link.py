"""add lighting-variant parent link on shot groups

Revision ID: e7a2b4c5d6f8
Revises: d8f4e6a1c2b3
Create Date: 2026-08-19 18:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'e7a2b4c5d6f8'
down_revision = 'd8f4e6a1c2b3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('shot_groups', schema=None) as batch_op:
        batch_op.add_column(sa.Column('parent_id', sa.Integer(), nullable=True))
        batch_op.create_foreign_key('fk_shot_groups_parent_id', 'shot_groups', ['parent_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    with op.batch_alter_table('shot_groups', schema=None) as batch_op:
        batch_op.drop_constraint('fk_shot_groups_parent_id', type_='foreignkey')
        batch_op.drop_column('parent_id')
