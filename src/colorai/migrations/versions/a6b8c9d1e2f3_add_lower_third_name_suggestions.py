"""add lower-third name suggestions and subject confirmation

Revision ID: a6b8c9d1e2f3
Revises: f3c5d7e9a1b2
Create Date: 2026-08-19 20:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


revision = 'a6b8c9d1e2f3'
down_revision = 'f3c5d7e9a1b2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table('subjects', schema=None) as batch_op:
        batch_op.add_column(sa.Column('name_confirmed', sa.Boolean(), nullable=False, server_default=sa.false()))

    op.create_table('name_suggestions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('asset_id', sa.Integer(), nullable=False),
        sa.Column('subject_id', sa.Integer(), nullable=True),
        sa.Column('shot_id', sa.Integer(), nullable=False),
        sa.Column('candidate_name', sa.String(length=255), nullable=False),
        sa.Column('raw_text', sa.Text(), nullable=False),
        sa.Column('role_text', sa.Text(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('timecode', sa.String(length=16), nullable=False),
        sa.Column('crop_path', sa.String(length=4096), nullable=True),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['asset_id'], ['media_assets.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['shot_id'], ['shots.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['subject_id'], ['subjects.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('name_suggestions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_name_suggestions_asset_id'), ['asset_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_name_suggestions_shot_id'), ['shot_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_name_suggestions_subject_id'), ['subject_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('name_suggestions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_name_suggestions_subject_id'))
        batch_op.drop_index(batch_op.f('ix_name_suggestions_shot_id'))
        batch_op.drop_index(batch_op.f('ix_name_suggestions_asset_id'))

    op.drop_table('name_suggestions')

    with op.batch_alter_table('subjects', schema=None) as batch_op:
        batch_op.drop_column('name_confirmed')
