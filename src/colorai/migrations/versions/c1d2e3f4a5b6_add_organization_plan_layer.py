"""add organization plan layer

Revision ID: c1d2e3f4a5b6
Revises: a6b8c9d1e2f3
Create Date: 2026-08-19 21:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'c1d2e3f4a5b6'
down_revision = 'a6b8c9d1e2f3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('organization_plans',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('author', sa.String(length=64), nullable=False),
        sa.Column('summary', sa.Text(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('approved_by', sa.String(length=64), nullable=True),
        sa.Column('approved_at', sa.DateTime(), nullable=True),
        sa.Column('applied_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['asset_id'], ['media_assets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('organization_plans', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_organization_plans_asset_id'), ['asset_id'], unique=False)

    op.create_table('organization_plan_groups',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=False),
        sa.Column('draft_key', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('kind', sa.String(length=16), nullable=False),
        sa.Column('camera', sa.String(length=64), nullable=True),
        sa.Column('parent_draft_key', sa.String(length=64), nullable=True),
        sa.Column('existing_group_id', sa.Integer(), nullable=True),
        sa.Column('participant_ids', sa.JSON(), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['existing_group_id'], ['shot_groups.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['plan_id'], ['organization_plans.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('plan_id', 'draft_key', name='uq_org_plan_group_draft_key')
    )
    with op.batch_alter_table('organization_plan_groups', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_organization_plan_groups_plan_id'), ['plan_id'], unique=False)

    op.create_table('organization_plan_items',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('plan_id', sa.Integer(), nullable=False),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('decision', sa.String(length=16), nullable=False),
        sa.Column('destination_type', sa.String(length=32), nullable=False),
        sa.Column('target_group_id', sa.Integer(), nullable=True),
        sa.Column('target_draft_key', sa.String(length=64), nullable=True),
        sa.Column('reason', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('evidence', sa.JSON(), nullable=True),
        sa.Column('human_override_reason', sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['organization_plans.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['target_group_id'], ['shot_groups.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('plan_id', 'shot_id', name='uq_org_plan_item_shot')
    )
    with op.batch_alter_table('organization_plan_items', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_organization_plan_items_plan_id'), ['plan_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_organization_plan_items_shot_id'), ['shot_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('organization_plan_items', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_organization_plan_items_shot_id'))
        batch_op.drop_index(batch_op.f('ix_organization_plan_items_plan_id'))

    op.drop_table('organization_plan_items')

    with op.batch_alter_table('organization_plan_groups', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_organization_plan_groups_plan_id'))

    op.drop_table('organization_plan_groups')

    with op.batch_alter_table('organization_plans', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_organization_plans_asset_id'))

    op.drop_table('organization_plans')
