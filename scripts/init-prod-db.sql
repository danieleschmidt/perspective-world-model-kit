-- Production database initialization script for PWMK
-- This script sets up the production database schema and initial data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS pwmk_core;
CREATE SCHEMA IF NOT EXISTS pwmk_experiments;
CREATE SCHEMA IF NOT EXISTS pwmk_monitoring;

-- Set search path
SET search_path TO pwmk_core, public;

-- Create tables for experiment tracking
CREATE TABLE IF NOT EXISTS experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    config JSONB NOT NULL,
    status VARCHAR(50) DEFAULT 'created',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    tags TEXT[],
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create table for model checkpoints
CREATE TABLE IF NOT EXISTS model_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    checkpoint_path VARCHAR(500) NOT NULL,
    epoch INTEGER NOT NULL,
    step BIGINT NOT NULL,
    metrics JSONB DEFAULT '{}'::jsonb,
    model_config JSONB DEFAULT '{}'::jsonb,
    file_size BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create table for training metrics
CREATE TABLE IF NOT EXISTS training_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    epoch INTEGER NOT NULL,
    step BIGINT NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Create table for belief states tracking
CREATE TABLE IF NOT EXISTS belief_states (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    timestep INTEGER NOT NULL,
    beliefs JSONB NOT NULL,
    confidence_scores JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create tables for environment episodes
CREATE TABLE IF NOT EXISTS episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(id) ON DELETE CASCADE,
    episode_number INTEGER NOT NULL,
    total_reward FLOAT,
    total_steps INTEGER,
    success BOOLEAN,
    environment_config JSONB DEFAULT '{}'::jsonb,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Create table for agent actions and observations
CREATE TABLE IF NOT EXISTS agent_trajectories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id UUID REFERENCES episodes(id) ON DELETE CASCADE,
    agent_id VARCHAR(100) NOT NULL,
    timestep INTEGER NOT NULL,
    observation JSONB NOT NULL,
    action JSONB NOT NULL,
    reward FLOAT NOT NULL,
    done BOOLEAN DEFAULT FALSE,
    info JSONB DEFAULT '{}'::jsonb
);

-- Create monitoring schema tables
SET search_path TO pwmk_monitoring, public;

-- Create table for system metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    labels JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create table for application logs
CREATE TABLE IF NOT EXISTS application_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    level VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    logger_name VARCHAR(100),
    module VARCHAR(100),
    function_name VARCHAR(100),
    line_number INTEGER,
    extra_data JSONB DEFAULT '{}'::jsonb,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
SET search_path TO pwmk_core, public;

-- Experiments indexes
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_experiments_tags ON experiments USING GIN(tags);

-- Model checkpoints indexes
CREATE INDEX IF NOT EXISTS idx_checkpoints_experiment ON model_checkpoints(experiment_id);
CREATE INDEX IF NOT EXISTS idx_checkpoints_epoch_step ON model_checkpoints(epoch, step);

-- Training metrics indexes
CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON training_metrics(experiment_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name_timestamp ON training_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON training_metrics(step);

-- Belief states indexes
CREATE INDEX IF NOT EXISTS idx_beliefs_experiment ON belief_states(experiment_id);
CREATE INDEX IF NOT EXISTS idx_beliefs_agent_timestep ON belief_states(agent_id, timestep);

-- Episodes indexes
CREATE INDEX IF NOT EXISTS idx_episodes_experiment ON episodes(experiment_id);
CREATE INDEX IF NOT EXISTS idx_episodes_number ON episodes(episode_number);

-- Trajectories indexes
CREATE INDEX IF NOT EXISTS idx_trajectories_episode ON agent_trajectories(episode_id);
CREATE INDEX IF NOT EXISTS idx_trajectories_agent_timestep ON agent_trajectories(agent_id, timestep);

-- Monitoring indexes
SET search_path TO pwmk_monitoring, public;

CREATE INDEX IF NOT EXISTS idx_system_metrics_name_timestamp ON system_metrics(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_system_metrics_labels ON system_metrics USING GIN(labels);
CREATE INDEX IF NOT EXISTS idx_logs_level_timestamp ON application_logs(level, timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_logger ON application_logs(logger_name);

-- Create functions for automatic timestamp updates
SET search_path TO pwmk_core, public;

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers
CREATE TRIGGER update_experiments_updated_at 
    BEFORE UPDATE ON experiments 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create views for common queries
CREATE OR REPLACE VIEW experiment_summary AS
SELECT 
    e.id,
    e.name,
    e.status,
    e.created_at,
    e.completed_at,
    COUNT(DISTINCT mc.id) as checkpoint_count,
    COUNT(DISTINCT ep.id) as episode_count,
    AVG(ep.total_reward) as avg_reward
FROM experiments e
LEFT JOIN model_checkpoints mc ON e.id = mc.experiment_id
LEFT JOIN episodes ep ON e.id = ep.experiment_id
GROUP BY e.id, e.name, e.status, e.created_at, e.completed_at;

CREATE OR REPLACE VIEW latest_metrics AS
SELECT DISTINCT ON (experiment_id, metric_name)
    experiment_id,
    metric_name,
    metric_value,
    timestamp
FROM training_metrics
ORDER BY experiment_id, metric_name, timestamp DESC;

-- Set up permissions for the application user
GRANT USAGE ON SCHEMA pwmk_core TO pwmk;
GRANT USAGE ON SCHEMA pwmk_experiments TO pwmk;
GRANT USAGE ON SCHEMA pwmk_monitoring TO pwmk;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pwmk_core TO pwmk;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pwmk_experiments TO pwmk;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA pwmk_monitoring TO pwmk;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pwmk_core TO pwmk;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pwmk_experiments TO pwmk;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA pwmk_monitoring TO pwmk;

-- Set default search path for the application user
ALTER USER pwmk SET search_path TO pwmk_core, pwmk_experiments, pwmk_monitoring, public;

-- Create initial data
INSERT INTO experiments (name, description, config, status) VALUES
('initial_setup', 'Initial system setup and validation', '{"version": "1.0.0"}', 'completed')
ON CONFLICT DO NOTHING;

-- Log successful initialization
INSERT INTO pwmk_monitoring.application_logs (level, message, logger_name, module) VALUES
('INFO', 'Production database initialized successfully', 'db_init', 'init_script');