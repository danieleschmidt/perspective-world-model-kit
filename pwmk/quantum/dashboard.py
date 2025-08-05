"""
Quantum algorithm monitoring dashboard and visualization.

Provides real-time dashboards and visualization capabilities for monitoring
quantum-enhanced planning performance and system health.
"""

from typing import Dict, List, Tuple, Optional, Any, Callable
import json
import time
from pathlib import Path
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

from ..utils.logging import LoggingMixin
from .monitoring import QuantumMetricsCollector, MetricType, QuantumDashboardConfig


@dataclass
class DashboardWidget:
    """Configuration for a dashboard widget."""
    widget_id: str
    widget_type: str  # "line_chart", "gauge", "bar_chart", "table", "heatmap"
    title: str
    metrics: List[str]
    refresh_interval: float = 1.0
    config: Dict[str, Any] = None


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    layout_id: str
    title: str
    widgets: List[DashboardWidget]
    columns: int = 2
    auto_refresh: bool = True
    theme: str = "dark"


class QuantumDashboard(LoggingMixin):
    """
    Real-time dashboard for quantum algorithm monitoring.
    
    Provides interactive visualization of quantum metrics, performance trends,
    and system health indicators with customizable layouts and widgets.
    """
    
    def __init__(
        self,
        metrics_collector: QuantumMetricsCollector,
        dashboard_config: Optional[QuantumDashboardConfig] = None,
        export_path: Optional[str] = None
    ):
        super().__init__()
        
        self.metrics_collector = metrics_collector
        self.config = dashboard_config or QuantumDashboardConfig()
        self.export_path = Path(export_path) if export_path else Path("quantum_dashboard")
        self.export_path.mkdir(exist_ok=True)
        
        # Dashboard state
        self.layouts = {}
        self.active_layout = None
        self.widget_data = defaultdict(dict)
        
        # Real-time update thread
        self.update_thread = None
        self.update_active = False
        
        # Dashboard templates
        self._initialize_default_layouts()
        
        self.logger.info(f"Initialized QuantumDashboard: export_path={self.export_path}")
    
    def _initialize_default_layouts(self) -> None:
        """Initialize default dashboard layouts."""
        
        # Performance overview layout
        performance_layout = DashboardLayout(
            layout_id="performance_overview",
            title="Quantum Performance Overview",
            widgets=[
                DashboardWidget(
                    widget_id="quantum_advantage_trend",
                    widget_type="line_chart",
                    title="Quantum Advantage Over Time",
                    metrics=["quantum_advantage"],
                    config={"y_range": [1.0, 5.0], "show_target": 2.0}
                ),
                DashboardWidget(
                    widget_id="planning_time_distribution",
                    widget_type="bar_chart",
                    title="Planning Time Distribution",
                    metrics=["planning_times"],
                    config={"bins": 20, "show_mean": True}
                ),
                DashboardWidget(
                    widget_id="system_resources",
                    widget_type="gauge",
                    title="System Resource Utilization",
                    metrics=["cpu_percent", "memory_percent", "gpu_memory_percent"],
                    config={"warning_threshold": 75, "critical_threshold": 90}
                ),
                DashboardWidget(
                    widget_id="algorithm_performance",
                    widget_type="table",
                    title="Algorithm Performance Summary",
                    metrics=["algorithm_stats"],
                    config={"max_rows": 10, "sortable": True}
                )
            ],
            columns=2
        )
        
        # Quantum circuit layout
        circuit_layout = DashboardLayout(
            layout_id="circuit_optimization",
            title="Quantum Circuit Optimization",
            widgets=[
                DashboardWidget(
                    widget_id="gate_count_reduction",
                    widget_type="line_chart",
                    title="Gate Count Reduction",
                    metrics=["gate_count_reduction"],
                    config={"y_range": [0.0, 1.0], "format": "percentage"}
                ),
                DashboardWidget(
                    widget_id="circuit_depth_trend",
                    widget_type="line_chart",
                    title="Circuit Depth Optimization",
                    metrics=["circuit_depth"],
                    config={"show_trend": True}
                ),
                DashboardWidget(
                    widget_id="fidelity_heatmap",
                    widget_type="heatmap",
                    title="Gate Fidelity Heat Map",
                    metrics=["gate_fidelity"],
                    config={"color_scale": "viridis", "min_value": 0.9}
                ),
                DashboardWidget(
                    widget_id="optimization_metrics",
                    widget_type="gauge",
                    title="Optimization Metrics",
                    metrics=["optimization_efficiency", "convergence_rate"],
                    config={"target_value": 0.85}
                )
            ],
            columns=2
        )
        
        # Annealing monitoring layout
        annealing_layout = DashboardLayout(
            layout_id="annealing_monitoring",
            title="Quantum Annealing Monitoring",
            widgets=[
                DashboardWidget(
                    widget_id="temperature_schedule",
                    widget_type="line_chart",
                    title="Annealing Temperature Schedule",
                    metrics=["temperature_schedule"],
                    config={"y_scale": "log", "show_phases": True}
                ),
                DashboardWidget(
                    widget_id="energy_convergence",
                    widget_type="line_chart",
                    title="Energy Convergence",
                    metrics=["energy_history"],
                    config={"show_convergence_point": True}
                ),
                DashboardWidget(
                    widget_id="tunneling_events",
                    widget_type="bar_chart",
                    title="Quantum Tunneling Events",
                    metrics=["tunneling_events"],
                    config={"color_by_value": True}
                ),
                DashboardWidget(
                    widget_id="annealing_success_rate",
                    widget_type="gauge",
                    title="Annealing Success Rate",
                    metrics=["success_rate"],
                    config={"target": 0.8, "format": "percentage"}
                )
            ],
            columns=2
        )
        
        # System health layout
        health_layout = DashboardLayout(
            layout_id="system_health",
            title="System Health & Diagnostics",
            widgets=[
                DashboardWidget(
                    widget_id="error_rate_trend",
                    widget_type="line_chart",
                    title="Error Rate Trend",
                    metrics=["error_rate"],
                    config={"alert_threshold": 0.05, "time_window": 300}
                ),
                DashboardWidget(
                    widget_id="cache_performance",
                    widget_type="gauge",
                    title="Cache Hit Rate",
                    metrics=["cache_hit_rate"],
                    config={"target": 0.8, "warning": 0.6}
                ),
                DashboardWidget(
                    widget_id="throughput_monitor",
                    widget_type="line_chart", 
                    title="Operations Throughput",
                    metrics=["ops_per_second"],
                    config={"show_moving_average": True, "window": 60}
                ),
                DashboardWidget(
                    widget_id="alert_summary",
                    widget_type="table",
                    title="Active Alerts",
                    metrics=["active_alerts"],
                    config={"highlight_critical": True, "auto_refresh": 5.0}
                )
            ],
            columns=2
        )
        
        # Register layouts
        self.layouts = {
            "performance_overview": performance_layout,
            "circuit_optimization": circuit_layout,
            "annealing_monitoring": annealing_layout,
            "system_health": health_layout
        }
        
        # Set default active layout
        self.active_layout = "performance_overview"
        
        self.logger.debug(f"Initialized {len(self.layouts)} default dashboard layouts")
    
    def start_real_time_updates(self) -> None:
        """Start real-time dashboard updates."""
        
        if self.update_active:
            self.logger.warning("Real-time updates already active")
            return
        
        self.update_active = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()
        
        self.logger.info("Started real-time dashboard updates")
    
    def stop_real_time_updates(self) -> None:
        """Stop real-time dashboard updates."""
        
        self.update_active = False
        
        if self.update_thread and self.update_thread.is_alive():
            self.update_thread.join(timeout=5.0)
        
        self.logger.info("Stopped real-time dashboard updates")
    
    def _update_loop(self) -> None:
        """Main update loop for real-time dashboard."""
        
        while self.update_active:
            try:
                # Update widget data for active layout
                if self.active_layout and self.active_layout in self.layouts:
                    layout = self.layouts[self.active_layout]
                    
                    for widget in layout.widgets:
                        self._update_widget_data(widget)
                
                # Sleep for update interval
                time.sleep(self.config.update_interval)
                
            except Exception as e:
                self.logger.error(f"Dashboard update loop error: {e}")
                time.sleep(1.0)  # Longer delay on error
    
    def _update_widget_data(self, widget: DashboardWidget) -> None:
        """Update data for a specific widget."""
        
        try:
            widget_data = {}
            current_time = time.time()
            
            # Get data based on widget type and metrics
            for metric_name in widget.metrics:
                if metric_name in ["quantum_advantage"]:
                    # Get quantum advantage trend
                    trend_data = self.metrics_collector.get_quantum_advantage_trend(100)
                    widget_data[metric_name] = {
                        "values": trend_data,
                        "timestamps": [current_time - i for i in range(len(trend_data), 0, -1)],
                        "current": trend_data[-1] if trend_data else 0.0
                    }
                
                elif metric_name in ["planning_times"]:
                    # Get planning time statistics
                    stats = self.metrics_collector.get_integration_statistics() if hasattr(self.metrics_collector, 'get_integration_statistics') else {}
                    planning_times = stats.get("planning_times", [])
                    
                    widget_data[metric_name] = {
                        "values": planning_times[-50:],  # Last 50 measurements
                        "mean": np.mean(planning_times) if planning_times else 0.0,
                        "std": np.std(planning_times) if planning_times else 0.0,
                        "min": np.min(planning_times) if planning_times else 0.0,
                        "max": np.max(planning_times) if planning_times else 0.0
                    }
                
                elif metric_name in ["cpu_percent", "memory_percent", "gpu_memory_percent"]:
                    # Get current resource metrics
                    try:
                        import psutil
                        if metric_name == "cpu_percent":
                            value = psutil.cpu_percent()
                        elif metric_name == "memory_percent":
                            value = psutil.virtual_memory().percent
                        else:  # gpu_memory_percent
                            value = 0.0  # Default if GPU not available
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    gpu_memory_used = torch.cuda.memory_allocated()
                                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                                    value = (gpu_memory_used / gpu_memory_total) * 100
                            except:
                                pass
                        
                        widget_data[metric_name] = {
                            "current": value,
                            "status": "normal" if value < 75 else "warning" if value < 90 else "critical"
                        }
                    except ImportError:
                        widget_data[metric_name] = {"current": 0.0, "status": "unknown"}
                
                elif metric_name in ["algorithm_stats"]:
                    # Get algorithm performance statistics
                    if hasattr(self.metrics_collector, 'algorithm_stats'):
                        stats = dict(self.metrics_collector.algorithm_stats)
                        widget_data[metric_name] = {
                            "algorithms": list(stats.keys()),
                            "data": stats,
                            "last_updated": max([s.get("last_updated", 0) for s in stats.values()]) if stats else 0
                        }
                    else:
                        widget_data[metric_name] = {"algorithms": [], "data": {}}
                
                elif metric_name in ["gate_count_reduction", "circuit_depth", "gate_fidelity"]:
                    # Get circuit optimization metrics
                    circuit_metrics = self.metrics_collector.get_circuit_optimization_metrics()
                    
                    if metric_name == "gate_count_reduction":
                        widget_data[metric_name] = {
                            "current": 0.0,  # Would need historical data
                            "trend": "stable"
                        }
                    elif metric_name == "circuit_depth":
                        depth_data = circuit_metrics.get("circuit_depth", {})
                        widget_data[metric_name] = {
                            "current": depth_data.get("current", 0),
                            "average": depth_data.get("average", 0),
                            "reduction_trend": depth_data.get("reduction_trend", 0.0)
                        }
                    else:  # gate_fidelity
                        fidelity_data = circuit_metrics.get("gate_fidelity", {})
                        widget_data[metric_name] = {
                            "current": fidelity_data.get("current", 0.0),
                            "average": fidelity_data.get("average", 0.0)
                        }
                
                elif metric_name in ["temperature_schedule", "energy_history", "tunneling_events", "success_rate"]:
                    # Get annealing performance metrics
                    annealing_metrics = self.metrics_collector.get_annealing_performance()
                    
                    if annealing_metrics.get("status") == "active":
                        if metric_name == "success_rate":
                            convergence_analysis = annealing_metrics.get("convergence_analysis", {})
                            widget_data[metric_name] = {
                                "current": convergence_analysis.get("success_rate", 0.0),
                                "target": 0.8
                            }
                        else:
                            widget_data[metric_name] = {
                                "data": [],  # Would need specific annealing run data
                                "status": "no_data"
                            }
                    else:
                        widget_data[metric_name] = {"status": "inactive"}
                
                else:
                    # Generic metric handling
                    widget_data[metric_name] = {
                        "value": 0.0,
                        "status": "unknown"
                    }
            
            # Store widget data
            self.widget_data[widget.widget_id] = {
                "data": widget_data,
                "last_updated": current_time,
                "widget_config": asdict(widget)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update widget {widget.widget_id}: {e}")
    
    def get_layout_data(self, layout_id: Optional[str] = None) -> Dict[str, Any]:
        """Get complete data for a dashboard layout."""
        
        if layout_id is None:
            layout_id = self.active_layout
        
        if layout_id not in self.layouts:
            raise ValueError(f"Layout {layout_id} not found")
        
        layout = self.layouts[layout_id]
        
        layout_data = {
            "layout": asdict(layout),
            "widgets": {},
            "timestamp": time.time()
        }
        
        # Get data for each widget
        for widget in layout.widgets:
            if widget.widget_id in self.widget_data:
                layout_data["widgets"][widget.widget_id] = self.widget_data[widget.widget_id]
            else:
                # Generate initial data
                self._update_widget_data(widget)
                layout_data["widgets"][widget.widget_id] = self.widget_data.get(widget.widget_id, {})
        
        return layout_data
    
    def export_dashboard_html(
        self, 
        layout_id: Optional[str] = None,
        filename: Optional[str] = None
    ) -> str:
        """Export dashboard as standalone HTML file."""
        
        layout_data = self.get_layout_data(layout_id)
        
        if filename is None:
            timestamp = int(time.time())
            layout_name = layout_id or self.active_layout
            filename = f"quantum_dashboard_{layout_name}_{timestamp}.html"
        
        html_file = self.export_path / filename
        
        # Generate HTML content
        html_content = self._generate_html_dashboard(layout_data)
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Exported dashboard to {html_file}")
        return str(html_file)
    
    def _generate_html_dashboard(self, layout_data: Dict[str, Any]) -> str:
        """Generate HTML content for dashboard."""
        
        layout = layout_data["layout"]
        widgets = layout_data["widgets"]
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{layout['title']} - Quantum Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            font-family: Arial, sans-serif;
            background-color: {'#1e1e1e' if layout.get('theme') == 'dark' else '#ffffff'};
            color: {'#ffffff' if layout.get('theme') == 'dark' else '#000000'};
        }}
        .dashboard-header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        .dashboard-grid {{
            display: grid;
            grid-template-columns: repeat({layout['columns']}, 1fr);
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .widget {{
            background: {'#2d2d2d' if layout.get('theme') == 'dark' else '#f5f5f5'};
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .widget-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            text-align: center;
        }}
        .widget-content {{
            height: 300px;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            padding: 20px;
        }}
        .metric-status-normal {{ color: #28a745; }}
        .metric-status-warning {{ color: #ffc107; }}
        .metric-status-critical {{ color: #dc3545; }}
        .timestamp {{
            text-align: center;
            margin-top: 20px;
            font-size: 12px;
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>{layout['title']}</h1>
        <p>Quantum Algorithm Monitoring Dashboard</p>
    </div>
    
    <div class="dashboard-grid">
        {self._generate_widget_html(widgets)}
    </div>
    
    <div class="timestamp">
        Last updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(layout_data['timestamp']))}
    </div>
    
    <script>
        {self._generate_dashboard_javascript(widgets)}
    </script>
</body>
</html>
        """
        
        return html_template
    
    def _generate_widget_html(self, widgets: Dict[str, Any]) -> str:
        """Generate HTML for dashboard widgets."""
        
        widget_html = []
        
        for widget_id, widget_data in widgets.items():
            widget_config = widget_data.get("widget_config", {})
            data = widget_data.get("data", {})
            
            widget_type = widget_config.get("widget_type", "table")
            title = widget_config.get("title", "Widget")
            
            if widget_type == "gauge":
                # Generate gauge widget
                html = f"""
                <div class="widget">
                    <div class="widget-title">{title}</div>
                    <div class="widget-content" id="{widget_id}">
                        {self._generate_gauge_content(data)}
                    </div>
                </div>
                """
            elif widget_type in ["line_chart", "bar_chart"]:
                # Generate chart widget
                html = f"""
                <div class="widget">
                    <div class="widget-title">{title}</div>
                    <div class="widget-content" id="{widget_id}"></div>
                </div>
                """
            elif widget_type == "table":
                # Generate table widget
                html = f"""
                <div class="widget">
                    <div class="widget-title">{title}</div>
                    <div class="widget-content" id="{widget_id}">
                        {self._generate_table_content(data)}
                    </div>
                </div>
                """
            else:
                # Default widget
                html = f"""
                <div class="widget">
                    <div class="widget-title">{title}</div>
                    <div class="widget-content" id="{widget_id}">
                        <p>Widget type '{widget_type}' not implemented</p>
                    </div>
                </div>
                """
            
            widget_html.append(html)
        
        return "\n".join(widget_html)
    
    def _generate_gauge_content(self, data: Dict[str, Any]) -> str:
        """Generate gauge widget content."""
        
        gauge_html = []
        
        for metric_name, metric_data in data.items():
            if isinstance(metric_data, dict) and "current" in metric_data:
                value = metric_data["current"]
                status = metric_data.get("status", "normal")
                
                gauge_html.append(f"""
                <div class="metric-value metric-status-{status}">
                    {metric_name.replace('_', ' ').title()}: {value:.1f}
                    {"%" if "percent" in metric_name else ""}
                </div>
                """)
        
        return "\n".join(gauge_html) if gauge_html else "<p>No data available</p>"
    
    def _generate_table_content(self, data: Dict[str, Any]) -> str:
        """Generate table content."""
        
        if not data:
            return "<p>No data available</p>"
        
        # Simple table generation
        table_html = "<table style='width:100%; border-collapse: collapse;'>"
        
        for key, value in data.items():
            if isinstance(value, dict):
                table_html += f"<tr><td style='border:1px solid #ccc; padding:8px;'><strong>{key}</strong></td><td style='border:1px solid #ccc; padding:8px;'>{len(value)} entries</td></tr>"
            else:
                table_html += f"<tr><td style='border:1px solid #ccc; padding:8px;'><strong>{key}</strong></td><td style='border:1px solid #ccc; padding:8px;'>{value}</td></tr>"
        
        table_html += "</table>"
        return table_html
    
    def _generate_dashboard_javascript(self, widgets: Dict[str, Any]) -> str:
        """Generate JavaScript for interactive dashboard."""
        
        js_code = []
        
        for widget_id, widget_data in widgets.items():
            widget_config = widget_data.get("widget_config", {})
            data = widget_data.get("data", {})
            widget_type = widget_config.get("widget_type", "table")
            
            if widget_type == "line_chart":
                js_code.append(self._generate_line_chart_js(widget_id, data, widget_config))
            elif widget_type == "bar_chart":
                js_code.append(self._generate_bar_chart_js(widget_id, data, widget_config))
        
        return "\n".join(js_code)
    
    def _generate_line_chart_js(self, widget_id: str, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate JavaScript for line chart."""
        
        js_template = f"""
        // Line chart for {widget_id}
        {{
            const data = [];
            const layout = {{
                title: '',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: 'white' }},
                margin: {{ t: 20, r: 20, b: 40, l: 40 }}
            }};
            
            // Add traces for each metric
            {self._generate_chart_data_js(data)}
            
            Plotly.newPlot('{widget_id}', data, layout, {{displayModeBar: false}});
        }}
        """
        
        return js_template
    
    def _generate_bar_chart_js(self, widget_id: str, data: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate JavaScript for bar chart."""
        
        js_template = f"""
        // Bar chart for {widget_id}
        {{
            const data = [];
            const layout = {{
                title: '',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
                font: {{ color: 'white' }},
                margin: {{ t: 20, r: 20, b: 40, l: 40 }}
            }};
            
            // Add bar chart data
            {self._generate_bar_data_js(data)}
            
            Plotly.newPlot('{widget_id}', data, layout, {{displayModeBar: false}});
        }}
        """
        
        return js_template
    
    def _generate_chart_data_js(self, data: Dict[str, Any]) -> str:
        """Generate JavaScript data for charts."""
        
        js_lines = []
        
        for metric_name, metric_data in data.items():
            if isinstance(metric_data, dict) and "values" in metric_data:
                values = metric_data["values"]
                timestamps = metric_data.get("timestamps", list(range(len(values))))
                
                js_lines.append(f"""
                data.push({{
                    x: {json.dumps(timestamps)},
                    y: {json.dumps(values)},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: '{metric_name.replace('_', ' ').title()}'
                }});
                """)
        
        return "\n".join(js_lines)
    
    def _generate_bar_data_js(self, data: Dict[str, Any]) -> str:
        """Generate JavaScript data for bar charts."""
        
        js_lines = []
        
        for metric_name, metric_data in data.items():
            if isinstance(metric_data, dict) and "values" in metric_data:
                values = metric_data["values"]
                
                # Create histogram bins
                if values:
                    js_lines.append(f"""
                    data.push({{
                        x: {json.dumps(values)},
                        type: 'histogram',
                        name: '{metric_name.replace('_', ' ').title()}',
                        nbinsx: 20
                    }});
                    """)
        
        return "\n".join(js_lines)
    
    def create_custom_layout(
        self,
        layout_id: str,
        title: str,
        widgets: List[DashboardWidget],
        columns: int = 2
    ) -> None:
        """Create a custom dashboard layout."""
        
        custom_layout = DashboardLayout(
            layout_id=layout_id,
            title=title,
            widgets=widgets,
            columns=columns
        )
        
        self.layouts[layout_id] = custom_layout
        self.logger.info(f"Created custom layout: {layout_id}")
    
    def switch_layout(self, layout_id: str) -> None:
        """Switch to a different dashboard layout."""
        
        if layout_id not in self.layouts:
            raise ValueError(f"Layout {layout_id} not found")
        
        self.active_layout = layout_id
        self.logger.info(f"Switched to layout: {layout_id}")
    
    def get_available_layouts(self) -> List[str]:
        """Get list of available dashboard layouts."""
        return list(self.layouts.keys())
    
    def export_layout_config(self, layout_id: str, filename: Optional[str] = None) -> str:
        """Export layout configuration to JSON file."""
        
        if layout_id not in self.layouts:
            raise ValueError(f"Layout {layout_id} not found")
        
        if filename is None:
            filename = f"layout_{layout_id}.json"
        
        config_file = self.export_path / filename
        layout_config = asdict(self.layouts[layout_id])
        
        with open(config_file, 'w') as f:
            json.dump(layout_config, f, indent=2)
        
        self.logger.info(f"Exported layout config to {config_file}")
        return str(config_file)
    
    def import_layout_config(self, config_file: str) -> str:
        """Import layout configuration from JSON file."""
        
        with open(config_file, 'r') as f:
            layout_config = json.load(f)
        
        # Reconstruct layout object
        widgets = [DashboardWidget(**w) for w in layout_config["widgets"]]
        layout = DashboardLayout(
            layout_id=layout_config["layout_id"],
            title=layout_config["title"],
            widgets=widgets,
            columns=layout_config.get("columns", 2),
            auto_refresh=layout_config.get("auto_refresh", True),
            theme=layout_config.get("theme", "dark")
        )
        
        self.layouts[layout.layout_id] = layout
        
        self.logger.info(f"Imported layout: {layout.layout_id}")
        return layout.layout_id
    
    def cleanup(self) -> None:
        """Clean up dashboard resources."""
        
        self.stop_real_time_updates()
        self.widget_data.clear()
        
        self.logger.info("Dashboard cleanup completed")