"""add editorial shot state and grouping

Revision ID: b5e2d9c3a4f6
Revises: c7d3e8a1f2b4
Create Date: 2026-08-19 15:10:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'b5e2d9c3a4f6'
down_revision = 'c7d3e8a1f2b4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('shot_groups',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['asset_id'], ['media_assets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('shot_groups', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_shot_groups_asset_id'), ['asset_id'], unique=False)

    with op.batch_alter_table('shots', schema=None) as batch_op:
        batch_op.add_column(sa.Column('review_status', sa.String(length=16), nullable=False, server_default='pending'))
        batch_op.add_column(sa.Column('excused', sa.Boolean(), nullable=False, server_default=sa.false()))
        batch_op.add_column(sa.Column('group_id', sa.Integer(), nullable=True))
        batch_op.create_index(batch_op.f('ix_shots_group_id'), ['group_id'], unique=False)
        batch_op.create_foreign_key('fk_shots_group_id', 'shot_groups', ['group_id'], ['id'], ondelete='SET NULL')


def downgrade() -> None:
    with op.batch_alter_table('shots', schema=None) as batch_op:
        batch_op.drop_constraint('fk_shots_group_id', type_='foreignkey')
        batch_op.drop_index(batch_op.f('ix_shots_group_id'))
        batch_op.drop_column('group_id')
        batch_op.drop_column('excused')
        batch_op.drop_column('review_status')

    with op.batch_alter_table('shot_groups', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_shot_groups_asset_id'))

    op.drop_table('shot_groups')
