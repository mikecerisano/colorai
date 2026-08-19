"""add reference proposals and setup-group fields

Revision ID: d8f4e6a1c2b3
Revises: b5e2d9c3a4f6
Create Date: 2026-08-19 17:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'd8f4e6a1c2b3'
down_revision = 'b5e2d9c3a4f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('shot_groups', schema=None) as batch_op:
        batch_op.add_column(sa.Column('kind', sa.String(length=16), nullable=False, server_default='generic'))
        batch_op.add_column(sa.Column('camera', sa.String(length=64), nullable=True))

    op.create_table('reference_proposals',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=True),
        sa.Column('group_id', sa.Integer(), nullable=True),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('author', sa.String(length=64), nullable=False),
        sa.Column('reason', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['asset_id'], ['media_assets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['group_id'], ['shot_groups.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('reference_proposals', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_reference_proposals_asset_id'), ['asset_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_reference_proposals_shot_id'), ['shot_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('reference_proposals', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_reference_proposals_shot_id'))
        batch_op.drop_index(batch_op.f('ix_reference_proposals_asset_id'))

    op.drop_table('reference_proposals')

    with op.batch_alter_table('shot_groups', schema=None) as batch_op:
        batch_op.drop_column('camera')
        batch_op.drop_column('kind')
