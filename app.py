"""
Sentiment Game Analytics Dashboard
===================================
A comprehensive dashboard for analyzing Steam game reviews sentiment.
Redesigned with features from previous Streamlit app.
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import re
import io
import base64

# WordCloud dependencies
from wordcloud import WordCloud
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for server
import matplotlib.pyplot as plt

# Sentiment prediction model dependencies
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ============================================
# ISSUE TAXONOMY FOR DEVELOPER INSIGHTS
# ============================================
ISSUE_TAXONOMY = {
    "🖥️ Technical": {
        "keywords": ["lag", "crash", "bug", "fps", "optimization", "server", "freeze", "loading", 
                     "performance", "stuttering", "glitch", "error", "disconnect", "memory", "frame"],
        "color": "#ff6b6b",  # Console Red
        "recommendations": {
            "high": "Consider infrastructure scaling, code profiling, or optimization patches",
            "medium": "Monitor error logs and prioritize critical fixes in next patch",
            "low": "Add to backlog for future optimization sprint"
        }
    },
    "🎮 Gameplay": {
        "keywords": ["balance", "difficulty", "controls", "mechanics", "combat", "boring", 
                     "repetitive", "grind", "unfair", "broken", "nerf", "buff", "skill", "enemy"],
        "color": "#4ecdc4",  # Secondary Teal
        "recommendations": {
            "high": "Conduct gameplay testing sessions and adjust difficulty curves",
            "medium": "Review player progression data and rebalance mechanics",
            "low": "Gather more feedback before making changes"
        }
    },
    "📖 Content": {
        "keywords": ["story", "dlc", "mission", "quest", "content", "short", "ending", 
                     "character", "update", "more", "expansion", "level", "world", "lore"],
        "color": "#4cc9f0",  # Console Blue
        "recommendations": {
            "high": "Prioritize content expansion in roadmap - strong revenue potential",
            "medium": "Plan seasonal content updates to maintain engagement",
            "low": "Consider community-driven content or modding support"
        }
    },
    "💰 Monetization": {
        "keywords": ["price", "expensive", "cheap", "worth", "money", "microtransaction", 
                     "p2w", "pay", "cost", "value", "purchase", "sale", "discount"],
        "color": "#ffbe0b",  # Console Amber
        "recommendations": {
            "high": "Review pricing strategy - consider regional pricing or bundles",
            "medium": "Improve value perception through better communication",
            "low": "Monitor competitor pricing and adjust accordingly"
        }
    }
}

# ============================================
# DATA LOADING & PREPROCESSING
# ============================================

def load_data():
    """Load and preprocess all data files."""
    df = pd.read_csv('data/steam_reviews.csv')
    
    sentiment_map = {'Positive': 'Satisfied', 'Negative': 'Dissatisfied', 'Neutral': 'Neutral'}
    df['sentiment_display'] = df['sentiment_name'].map(sentiment_map)
    df['sentiment'] = df['sentiment_name'].map({'Positive': 2, 'Neutral': 1, 'Negative': 0})
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['month'] = df['timestamp'].dt.to_period('M').astype(str)
    
    try:
        benchmark_df = pd.read_csv('data/benchmark.csv')
    except:
        benchmark_df = pd.DataFrame()
    
    try:
        confusion_df = pd.read_csv('data/roberta_confusion_matrix_data.csv', index_col=0)
    except:
        confusion_df = pd.DataFrame()
    
    try:
        training_loss_df = pd.read_csv('data/roberta_training_loss.csv')
    except:
        training_loss_df = pd.DataFrame()
    
    return df, benchmark_df, confusion_df, training_loss_df


def categorize_issues(text):
    """Categorize review text into issue categories."""
    if not isinstance(text, str):
        return 'Other'
    text_lower = text.lower()
    
    # Map emoji categories to clean names for display
    category_map = {
        "🖥️ Technical": "Technical",
        "🎮 Gameplay": "Gameplay",
        "📖 Content": "Content",
        "💰 Monetization": "Monetization"
    }
    
    for category, data in ISSUE_TAXONOMY.items():
        if any(keyword in text_lower for keyword in data['keywords']):
            return category_map.get(category, category)
    return 'Other'


def extract_issues(df_subset):
    """Extract and count issues by category."""
    issue_counts = {}
    for category in ISSUE_TAXONOMY.keys():
        keywords = ISSUE_TAXONOMY[category]["keywords"]
        count = df_subset['cleaned_text'].str.lower().str.contains('|'.join(keywords), na=False).sum()
        issue_counts[category] = count
    return issue_counts


def calculate_priority_matrix(issue_counts, df_subset):
    """Calculate data for Priority Matrix (Impact vs Frequency)."""
    matrix_data = []
    # Clean category names (remove emojis)
    clean_names = {
        "🖥️ Technical": "Technical",
        "🎮 Gameplay": "Gameplay",
        "📖 Content": "Content",
        "💰 Monetization": "Monetization"
    }
    
    for category, count in issue_counts.items():
        keywords = ISSUE_TAXONOMY[category]["keywords"]
        mask = df_subset['cleaned_text'].str.lower().str.contains('|'.join(keywords), na=False)
        subset = df_subset[mask]
        
        if len(subset) > 0:
            neg_ratio = (subset['sentiment'] == 0).mean()
            impact = neg_ratio * 100
        else:
            impact = 0
        
        matrix_data.append({
            "Category": clean_names.get(category, category),
            "Frequency": count,
            "Impact": impact,
            "Color": ISSUE_TAXONOMY[category]["color"]
        })
    return pd.DataFrame(matrix_data)


def generate_recommendations(issue_counts, total_reviews):
    """Generate AI-powered recommendations based on issue patterns."""
    recommendations = {"high": [], "medium": [], "low": []}
    
    for category, count in issue_counts.items():
        if total_reviews == 0:
            continue
        percentage = (count / total_reviews) * 100
        data = ISSUE_TAXONOMY[category]
        
        if percentage >= 15:
            priority = "high"
        elif percentage >= 5:
            priority = "medium"
        elif percentage >= 2:
            priority = "low"
        else:
            continue
        
        recommendations[priority].append({
            "category": category,
            "percentage": percentage,
            "count": count,
            "action": data["recommendations"][priority],
            "color": data["color"]
        })
    
    return recommendations


def get_player_quotes(df_subset, category, n=4):
    """Get representative quotes for a category."""
    keywords = ISSUE_TAXONOMY[category]["keywords"]
    mask = df_subset['cleaned_text'].str.lower().str.contains('|'.join(keywords), na=False)
    subset = df_subset[mask].copy()
    
    if subset.empty:
        return []
    
    subset = subset.sort_values('sentiment', ascending=True)
    quotes = []
    for _, row in subset.head(n).iterrows():
        text = str(row.get('cleaned_text', ''))[:200]
        sentiment = row.get('sentiment', 1)
        sentiment_label = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}.get(sentiment, "Unknown")
        quotes.append({
            "text": text,
            "sentiment": sentiment_label,
            "playtime": row.get('playtime_hours', 0),
            "helpful": row.get('helpful_votes', 0)
        })
    return quotes


# Load data
df, benchmark_df, confusion_df, training_loss_df = load_data()
df['issue_category'] = df['cleaned_text'].apply(categorize_issues)

# ============================================
# AI MODEL LOADING (HuggingFace RoBERTa)
# ============================================

# Cache the model to avoid reloading on each request
_model_cache = {'tokenizer': None, 'model': None, 'loaded': False, 'loading': False, 'error': None}

def load_sentiment_model():
    """Load the HuggingFace sentiment model (cached)"""
    if _model_cache['loaded']:
        return _model_cache['tokenizer'], _model_cache['model']
    
    if _model_cache['loading']:
        return None, None  # Still loading
    
    _model_cache['loading'] = True
    try:
        print("🤖 Loading AI model from HuggingFace...")
        _model_cache['tokenizer'] = AutoTokenizer.from_pretrained("manchae86/steam-review-roberta")
        _model_cache['model'] = AutoModelForSequenceClassification.from_pretrained("manchae86/steam-review-roberta")
        _model_cache['loaded'] = True
        _model_cache['loading'] = False
        print("✅ AI model loaded successfully!")
    except Exception as e:
        _model_cache['error'] = str(e)
        _model_cache['loading'] = False
        print(f"❌ Error loading model: {e}")
        return None, None
    return _model_cache['tokenizer'], _model_cache['model']

def predict_sentiment(text):
    """Predict sentiment for given text"""
    # Check if model is still loading
    if _model_cache['loading']:
        return None, None, "Model is still loading, please wait..."
    
    if _model_cache['error']:
        return None, None, f"Model loading failed: {_model_cache['error']}"
    
    tokenizer, model = load_sentiment_model()
    if tokenizer is None or model is None:
        return None, None, "Model not available. Please try again later."
    
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        score = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][score].item()
        
        labels = {0: "Dissatisfied", 1: "Neutral", 2: "Satisfied"}
        return labels[score], confidence, None
    except Exception as e:
        return None, None, str(e)

# Note: Model will be loaded on first use (lazy loading) to conserve memory on free tier

# ============================================
# COLOR SCHEME & TEMPLATE - RETRO PIXEL THEME
# ============================================

COLORS = {
    'Satisfied': '#92f29c',      # Console Green
    'Neutral': '#ffbe0b',        # Console Amber
    'Dissatisfied': '#ff6b6b',   # Console Red/Primary
    'bg': '#1a1a2e',             # Dark background
    'surface': '#1a1a2e',        # Same as bg for cards
    'text': '#f8f9fa',           # Light text
    'muted': '#ffffff',          # White for readability
    'accent': '#ff6b6b',         # Primary red
    'border': '#f8f9fa'          # Light border
}

SENTIMENT_COLORS = ['#ff6b6b', '#ffbe0b', '#92f29c']  # Red, Amber, Green

plotly_template = go.layout.Template()
plotly_template.layout = go.Layout(
    paper_bgcolor='#1a1a2e',  # Dark background to prevent white flash
    plot_bgcolor='#1a1a2e',   # Dark background to prevent white flash
    font=dict(family='VT323, monospace', color='#ffffff', size=18),
    title=dict(font=dict(family='VT323, monospace', size=20, color='#ffffff')),
    xaxis=dict(gridcolor='rgba(255,255,255,0.15)', zerolinecolor='rgba(255,255,255,0.3)', tickfont=dict(family='VT323, monospace', size=18, color='#ffffff')),
    yaxis=dict(gridcolor='rgba(255,255,255,0.15)', zerolinecolor='rgba(255,255,255,0.3)', tickfont=dict(family='VT323, monospace', size=18, color='#ffffff')),
    colorway=['#ff6b6b', '#92f29c', '#ffbe0b', '#4ecdc4', '#4cc9f0'],
    margin=dict(l=70, r=40, t=50, b=70),
    height=350,
    legend=dict(font=dict(family='VT323, monospace', size=18, color='#ffffff'))
)

# Chart interaction config - Simplified modebar (Download PNG only)
CHART_CONFIG = {
    'displayModeBar': 'hover',  # Show on hover
    'displaylogo': False,       # Hide Plotly logo
    'scrollZoom': False,        # Disable scroll zoom for easier page scrolling
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 
        'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d',
        'hoverClosestCartesian', 'hoverCompareCartesian',
        'toggleSpikelines'
    ],
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'sentiment_chart',
        'height': 600,
        'width': 1000,
        'scale': 2
    }
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def create_metric_card(value, label, icon=""):
    return html.Div([
        html.Div(f"{icon} {value}", className="metric-value"),
        html.Div(label, className="metric-label")
    ], className="metric-card")


def create_insight_card(category, percentage, count, color):
    return html.Div([
        html.Div(category, className="insight-header"),
        html.Div(f"{percentage:.1f}%", className="insight-value", style={'color': color}),
        html.Div(f"{count:,} mentions", style={'fontSize': '16px', 'color': '#f8f9fa', 'fontFamily': 'VT323, monospace', 'fontStyle': 'italic'})
    ], className="insight-card", style={'borderTop': f'3px solid {color}'})


def create_recommendation_card(rec, priority_class):
    return html.Div([
        html.Strong(rec['category']),
        html.Span(f" — {rec['percentage']:.1f}% ({rec['count']:,} mentions)", style={'fontSize': '16px', 'color': '#f8f9fa', 'fontFamily': 'VT323, monospace', 'fontStyle': 'italic'}),
        html.Br(),
        html.Span(f"→ {rec['action']}", style={'fontSize': '16px', 'color': '#f8f9fa', 'fontFamily': 'VT323, monospace', 'fontStyle': 'italic'})
    ], className=f"recommendation-card {priority_class}")


def create_section_hint(text):
    """Create a subtle hint text under section titles to help users understand the content."""
    return html.P(text, style={
        'fontSize': '16px',
        'color': '#f8f9fa',
        'marginBottom': '10px',
        'textAlign': 'center',
        'fontFamily': 'VT323, monospace',
        'fontStyle': 'italic'
    })


def create_quote_card(quote):
    sentiment_class = quote['sentiment'].lower()
    return html.Div([
        html.Div(f'"{quote["text"]}..."', className="quote-text"),
        html.Div([
            html.Span(f"😊 {quote['sentiment']}" if quote['sentiment'] == 'Satisfied' else 
                     f"😐 {quote['sentiment']}" if quote['sentiment'] == 'Neutral' else 
                     f"😡 {quote['sentiment']}"),
            html.Span(f" • {quote['playtime']:.0f}h playtime • {quote['helpful']} helpful votes")
        ], className="quote-meta")
    ], className=f"quote-card {sentiment_class}")


# ============================================
# APP INITIALIZATION
# ============================================

app = dash.Dash(
    __name__, 
    suppress_callback_exceptions=True,
    meta_tags=[
        # Mobile responsive viewport
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no"},
        # iOS web app settings
        {"name": "apple-mobile-web-app-capable", "content": "yes"},
        {"name": "apple-mobile-web-app-status-bar-style", "content": "black-translucent"},
        # Theme color for mobile browsers
        {"name": "theme-color", "content": "#1a1a2e"},
        # Description for SEO
        {"name": "description", "content": "AI-Powered Steam Game Reviews Sentiment Analysis Dashboard"}
    ]
)
app.title = "Sentiment Game Analytics"

# ============================================
# TAB LAYOUTS
# ============================================

def create_header():
    return html.Div([
        html.H1("🎮 Sentiment Game Analytics", className="dashboard-title"),
        html.P("AI-Powered Steam Reviews Analysis Dashboard", className="dashboard-subtitle")
    ], className="dashboard-header")


def create_tab1_dashboard():
    """Tab 1: Overview Dashboard"""
    total_reviews = len(df)
    satisfaction_rate = (df['sentiment_display'] == 'Satisfied').mean() * 100
    avg_playtime = df['playtime_hours'].mean()
    total_games = df['game_name'].nunique()
    
    # Sentiment donut
    sentiment_counts = df['sentiment_display'].value_counts()
    # Use dark text for light colors (green/amber) and white for dark colors
    text_colors = []
    for s in sentiment_counts.index:
        if s in ['Satisfied', 'Neutral']:  # Light backgrounds need dark text
            text_colors.append('#1a1a2e')
        else:
            text_colors.append('#ffffff')
    
    fig_donut = go.Figure(data=[go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.65,
        marker=dict(colors=[COLORS.get(s, COLORS['accent']) for s in sentiment_counts.index]),
        textinfo='percent+label',
        textfont=dict(size=14, color=text_colors),
        textposition='inside',
        insidetextorientation='radial'
    )])
    fig_donut.update_layout(
        template=plotly_template, showlegend=False, height=320,
        margin=dict(t=20, b=20, l=20, r=20),
        uniformtext=dict(minsize=10, mode='hide'),
        annotations=[dict(text='All Games', x=0.5, y=0.5, font_size=16, showarrow=False, font_color=COLORS['text'])]
    )
    
    # Trend over time - Format month for cleaner display
    trend_df = df.groupby(['month', 'sentiment_display']).size().reset_index(name='count')
    fig_trend = px.area(trend_df, x='month', y='count', color='sentiment_display',
                        color_discrete_map=COLORS, template=plotly_template)
    fig_trend.update_layout(
        xaxis_title="", yaxis_title="Reviews", height=320,
        margin=dict(l=60, r=40, t=50, b=70),
        xaxis=dict(tickangle=-45, nticks=6),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title=None)
    )
    
    # Game leaderboard
    game_sat = df.groupby('game_name').apply(lambda x: (x['sentiment'] == 2).mean()).reset_index(name='sat_rate')
    game_sat = game_sat.sort_values('sat_rate', ascending=True)
    fig_leaderboard = px.bar(game_sat, y='game_name', x='sat_rate', orientation='h',
                             template=plotly_template, text_auto='.0%',
                             color='sat_rate', color_continuous_scale=['#dc2626', '#eab308', '#22c55e'])
    fig_leaderboard.update_layout(xaxis_title="Satisfaction Rate", yaxis_title="", height=350,
                                  coloraxis_showscale=False, xaxis_tickformat='.0%',
                                  margin=dict(l=150, r=40, t=40, b=40))
    fig_leaderboard.update_traces(textposition='inside', textfont_color='white')
    
    return html.Div([
        html.Div([
            create_metric_card(f"{total_reviews:,}", "Total Reviews", "📊"),
            create_metric_card(f"{satisfaction_rate:.1f}%", "Satisfaction Rate", "😊"),
            create_metric_card(f"{avg_playtime:.0f}h", "Avg. Playtime", "⏱️"),
            create_metric_card(f"{total_games}", "Games Analyzed", "🎮"),
        ], className="metrics-grid"),
        
        html.Div([
            html.Div([
                html.H3("Sentiment Distribution", className="card-title"),
                create_section_hint("Overall player satisfaction breakdown across all games"),
                html.Div(f"Total: {total_reviews:,} reviews", style={'fontSize': '16px', 'color': COLORS['text'], 'marginBottom': '10px', 'textAlign': 'center', 'fontFamily': 'VT323, monospace'}),
                dcc.Graph(figure=fig_donut, config=CHART_CONFIG)
            ], className="card"),
            html.Div([
                html.H3("Sentiment Trends Over Time", className="card-title"),
                create_section_hint("Track how player sentiment changes month by month"),
                dcc.Graph(figure=fig_trend, config=CHART_CONFIG)
            ], className="card"),
        ], className="charts-grid"),
        
        html.Div([
            html.H3("🏆 Game Satisfaction Leaderboard", className="card-title"),
            create_section_hint("Compare satisfaction rates across all games • Hover over bars for details"),
            dcc.Graph(figure=fig_leaderboard, config=CHART_CONFIG)
        ], className="card full-width"),
    ])


def create_tab2_dev_insights(theme='dark'):
    """Tab 2: Developer Insights with Priority Matrix"""
    games = df['game_name'].unique()
    
    return html.Div([
        html.Div([
            html.Label("Select Game for Analysis:", style={'color': COLORS['muted'], 'fontSize': '0.875rem'}),
            dcc.Dropdown(
                id='dev-game-selector',
                options=[{'label': g, 'value': g} for g in games],
                value=games[0],
                clearable=False
            )
        ], style={'marginBottom': '1.25rem'}),
        
        # Executive Summary
        html.Div([
            html.H3("📊 Executive Summary", className="section-header"),
            create_section_hint("Quick overview of key issue categories detected in player reviews"),
            html.Div(id='issue-summary-cards', className="charts-grid-4"),
        ]),
        
        # Priority Matrix
        html.Div([
            html.H3("🎯 Priority Matrix: Impact vs Frequency", className="section-header"),
            html.P("Issues in the top-right quadrant (high frequency + high impact) need immediate attention.", 
                   style={'color': COLORS['muted'], 'fontSize': '0.875rem', 'marginBottom': '0.75rem'}),
            dcc.Graph(id='priority-matrix', config=CHART_CONFIG)
        ], className="card", style={'marginBottom': '1.25rem'}),
        
        # Recommendations
        html.Div([
            html.Div([
                html.H3("💡 AI-Powered Recommendations", className="section-header"),
                create_section_hint("Actionable suggestions based on issue analysis"),
                html.Div(id='recommendations-panel')
            ], className="card"),
            html.Div([
                html.H3("🗣️ Voice of the Player", className="section-header"),
                create_section_hint("Real player quotes by issue category"),
                html.Div([
                    html.Label("Issue Category:", style={'fontSize': '0.75rem', 'color': COLORS['muted']}),
                    dcc.Dropdown(
                        id='quote-category-selector',
                        options=[{'label': cat, 'value': cat} for cat in ISSUE_TAXONOMY.keys()],
                        value=list(ISSUE_TAXONOMY.keys())[0],
                        clearable=False,
                        style={'marginBottom': '0.75rem'}
                    ),
                ]),
                html.Div(id='player-quotes')
            ], className="card"),
        ], className="charts-grid"),
        
        # Issue Trend
        html.Div([
            html.H3("📈 Issue Trend Over Time", className="section-header"),
            create_section_hint("See how different issues spike or decline over months"),
            dcc.Graph(id='issue-trend-chart', config=CHART_CONFIG)
        ], className="card full-width"),
    ])


def create_tab3_comparison(theme='dark'):
    """Tab 3: Game Comparison Lab"""
    games = df['game_name'].unique().tolist()
    
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Game A:", style={'fontSize': '0.75rem', 'color': COLORS['muted']}),
                dcc.Dropdown(id='compare-game-a', options=[{'label': g, 'value': g} for g in games],
                            value=games[0] if len(games) > 0 else None, clearable=False)
            ], style={'flex': 1}),
            html.Div([
                html.Label("Game B:", style={'fontSize': '0.75rem', 'color': COLORS['muted']}),
                dcc.Dropdown(id='compare-game-b', options=[{'label': g, 'value': g} for g in games],
                            value=games[1] if len(games) > 1 else games[0], clearable=False)
            ], style={'flex': 1}),
        ], className="filter-bar"),
        create_section_hint("Select two games to compare their sentiment and issue patterns side-by-side"),
        
        html.Div([
            html.Div([
                html.H3(id='compare-title-a', className="card-title", style={'textAlign': 'center'}),
                dcc.Graph(id='compare-pie-a', config=CHART_CONFIG),
                html.Div(id='compare-stats-a', style={'textAlign': 'center'})
            ], className="card"),
            html.Div([
                html.H3(id='compare-title-b', className="card-title", style={'textAlign': 'center'}),
                dcc.Graph(id='compare-pie-b', config=CHART_CONFIG),
                html.Div(id='compare-stats-b', style={'textAlign': 'center'})
            ], className="card"),
        ], className="charts-grid"),
        
        html.Div([
            html.H3("🔍 Issue Category Comparison", className="card-title"),
            create_section_hint("Compare which issues are more prevalent in each game"),
            dcc.Graph(id='compare-issues-chart', config=CHART_CONFIG)
        ], className="card full-width"),
        
        html.Div(id='comparison-winner', style={'marginTop': '1rem'})
    ])


def create_tab4_explorer(theme='dark'):
    """Tab 4: Review Explorer"""
    games = ["All Games"] + df['game_name'].unique().tolist()
    
    return html.Div([
        html.Div([
            html.Div([
                html.Label("🔍 Search:", className="filter-label"),
                dcc.Input(id='explorer-search', type='text', placeholder='Enter keywords...',
                         style={'width': '100%'})
            ], className="filter-item", style={'flex': 2}),
            html.Div([
                html.Label("Sentiment:", className="filter-label"),
                dcc.Dropdown(id='explorer-sentiment', 
                            options=[{'label': s, 'value': s} for s in ['All', 'Satisfied', 'Neutral', 'Dissatisfied']],
                            value='All', clearable=False)
            ], className="filter-item"),
            html.Div([
                html.Label("Game:", className="filter-label"),
                dcc.Dropdown(id='explorer-game', options=[{'label': g, 'value': g} for g in games],
                            value='All Games', clearable=False)
            ], className="filter-item"),
        ], className="filter-bar"),
        create_section_hint("Search and filter through individual player reviews • Use keywords to find specific topics"),
        
        html.Div(id='explorer-count', style={'marginBottom': '0.75rem', 'color': COLORS['muted'], 'fontSize': '0.875rem'}),
        html.Div(id='explorer-results'),
        
        html.Div([
            html.Button("← Previous", id='explorer-prev', n_clicks=0, 
                       style={'padding': '0.5rem 1rem', 'marginRight': '0.5rem', 'cursor': 'pointer'}),
            html.Span(id='explorer-page-info'),
            html.Button("Next →", id='explorer-next', n_clicks=0,
                       style={'padding': '0.5rem 1rem', 'marginLeft': '0.5rem', 'cursor': 'pointer'}),
        ], className="pagination"),
        
        dcc.Store(id='explorer-page', data=1),
        dcc.Store(id='explorer-filtered-count', data=0)
    ])


def create_tab5_model():
    """Tab 5: Model Internals"""
    components = []
    
    # Model info cards
    components.append(html.Div([
        html.Div([
            html.Div("Base Architecture", className="insight-header"),
            html.Div("RoBERTa-Base", className="insight-value", style={'color': COLORS['accent'], 'fontSize': '1.25rem'}),
            html.Div("12-Layer Transformer", style={'fontSize': '16px', 'color': '#f8f9fa', 'fontFamily': 'VT323, monospace', 'fontStyle': 'italic'})
        ], className="insight-card"),
        html.Div([
            html.Div("Optimizer", className="insight-header"),
            html.Div("AdamW", className="insight-value", style={'color': COLORS['accent'], 'fontSize': '1.25rem'}),
            html.Div("Learning Rate: 2e-5", style={'fontSize': '16px', 'color': '#f8f9fa', 'fontFamily': 'VT323, monospace', 'fontStyle': 'italic'})
        ], className="insight-card"),
        html.Div([
            html.Div("Training Scale", className="insight-header"),
            html.Div(f"{len(df):,}", className="insight-value", style={'color': COLORS['accent'], 'fontSize': '1.25rem'}),
            html.Div("Reviews Analyzed", style={'fontSize': '16px', 'color': '#f8f9fa', 'fontFamily': 'VT323, monospace', 'fontStyle': 'italic'})
        ], className="insight-card"),
    ], className="charts-grid-3", style={'marginBottom': '1.25rem'}))
    
    # Confusion Matrix
    if not confusion_df.empty:
        confusion_display = confusion_df.copy()
        confusion_display.index = ['Dissatisfied', 'Neutral', 'Satisfied']
        confusion_display.columns = ['Dissatisfied', 'Neutral', 'Satisfied']
        
        fig_cm = px.imshow(
            confusion_display.values,
            x=confusion_display.columns,
            y=confusion_display.index,
            color_continuous_scale=['#f8fafc', '#93c5fd', '#3b82f6'],
            text_auto=True,
            template=plotly_template
        )
        fig_cm.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual", height=340,
            margin=dict(l=100, r=40, t=40, b=80),
            xaxis=dict(side='bottom', tickangle=0)
        )
        fig_cm.update_coloraxes(showscale=False)
    else:
        fig_cm = go.Figure()
    
    # Training Loss
    if not training_loss_df.empty:
        fig_loss = px.line(training_loss_df, x='step', y='loss', template=plotly_template,
                          color_discrete_sequence=[COLORS['accent']], markers=False)
        fig_loss.update_layout(xaxis_title="Training Step", yaxis_title="Loss", height=320)
        fig_loss.update_traces(line=dict(width=2))
    else:
        fig_loss = go.Figure()
    
    components.append(html.Div([
        html.Div([
            html.H3("🎯 Confusion Matrix", className="card-title"),
            create_section_hint("Model prediction accuracy across sentiment categories"),
            dcc.Graph(figure=fig_cm, config=CHART_CONFIG)
        ], className="card"),
        html.Div([
            html.H3("📉 Training Loss Curve", className="card-title"),
            create_section_hint("How the model improved during fine-tuning"),
            dcc.Graph(figure=fig_loss, config=CHART_CONFIG)
        ], className="card"),
    ], className="charts-grid"))
    
    # Benchmark
    if not benchmark_df.empty:
        bench_sorted = benchmark_df.sort_values("Accuracy", ascending=True)
        fig_bench = px.bar(bench_sorted, x="Accuracy", y="Model", orientation='h', text_auto='.1%',
                          color="Accuracy", color_continuous_scale=["#dc2626", "#eab308", "#22c55e"],
                          template=plotly_template)
        fig_bench.update_layout(xaxis_title="Accuracy", yaxis_title="", height=280, 
                               xaxis_tickformat='.0%', coloraxis_showscale=False,
                               margin=dict(l=120, r=40, t=40, b=40))
        fig_bench.update_traces(textposition='inside', textfont_color='white')
        
        components.append(html.Div([
            html.H3("🏆 Model Benchmark Comparison", className="card-title"),
            create_section_hint("Our fine-tuned model vs. other sentiment analysis approaches"),
            dcc.Graph(figure=fig_bench, config=CHART_CONFIG)
        ], className="card full-width"))
    
    return html.Div(components)


def create_tab6_wordcloud():
    """Tab 6: Topic Cloud/WordCloud Visualization"""
    games = ["All Games"] + df['game_name'].unique().tolist()
    
    return html.Div([
        # Settings Panel
        html.Div([
            html.Div([
                html.H3("☁️ Topic Cloud Settings", className="card-title"),
                create_section_hint("Visualize what players are talking about"),
                
                # Sentiment Filter
                html.Div([
                    html.Label("Sentiment Focus:", style={'fontWeight': '600', 'marginBottom': '1rem', 'display': 'block', 'fontFamily': 'VT323, monospace', 'fontSize': '18px'}),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span("😊", style={'fontSize': '28px'}),
                                html.Div("Satisfied", style={'fontFamily': 'VT323, monospace', 'fontSize': '16px', 'marginTop': '4px'}),
                            ], style={'textAlign': 'center'})
                        ], id='wc-btn-satisfied', className='sentiment-toggle-btn active', n_clicks=0),
                        html.Div([
                            html.Div([
                                html.Span("😡", style={'fontSize': '28px'}),
                                html.Div("Dissatisfied", style={'fontFamily': 'VT323, monospace', 'fontSize': '16px', 'marginTop': '4px'}),
                            ], style={'textAlign': 'center'})
                        ], id='wc-btn-dissatisfied', className='sentiment-toggle-btn', n_clicks=0),
                    ], style={'display': 'flex', 'gap': '0.75rem'}),
                    dcc.Store(id='wc-sentiment-filter', data='Satisfied'),
                ], style={'marginBottom': '1.5rem'}),
                
                # Game Filter
                html.Div([
                    html.Label("Game Filter:", style={'fontWeight': '600', 'marginBottom': '0.5rem', 'display': 'block', 'fontFamily': 'VT323, monospace', 'fontSize': '18px'}),
                    dcc.Dropdown(
                        id='wc-game-filter',
                        options=[{'label': g, 'value': g} for g in games],
                        value='All Games',
                        className='custom-dropdown',
                        clearable=False
                    ),
                ], style={'marginBottom': '1rem'}),
                
                html.Div([
                    html.Span("💡 ", style={'fontSize': '20px'}),
                    html.Span("Pro Tip: Common words like 'game', 'play', 'steam' are filtered out", 
                             style={'color': COLORS['muted'], 'fontSize': '14px', 'fontFamily': 'VT323, monospace'})
                ], className="insight-hint")
            ], className="card", style={'height': 'fit-content'}),
        ], style={'flex': '1', 'minWidth': '280px'}),
        
        # Word Cloud Display
        html.Div([
            html.Div([
                html.H3("📊 Word Cloud Visualization", className="card-title"),
                html.Div(id='wordcloud-container', style={'minHeight': '400px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
            ], className="card", style={'height': 'fit-content'}),
        ], style={'flex': '2', 'minWidth': '400px'}),
    ], style={'display': 'flex', 'gap': '1.5rem', 'flexWrap': 'wrap'})


def create_tab7_live_ai():
    """Tab 7: Real-time Sentiment Detection / Live AI Lab"""
    return html.Div([
        html.Div([
            html.H3("🤖 Real-time Sentiment Detection", className="card-title"),
            create_section_hint("Test the AI model with your own custom text"),
        ], style={'marginBottom': '1.5rem'}),
        
        html.Div([
            # Input Section
            html.Div([
                html.Div([
                    html.H4("✍️ Enter Your Review", style={'color': COLORS['text'], 'marginBottom': '1rem', 'fontFamily': 'VT323, monospace', 'fontSize': '22px'}),
                    dcc.Textarea(
                        id='ai-text-input',
                        placeholder='Example: The combat is peak but the optimization is trash...',
                        style={
                            'width': '100%',
                            'height': '200px',
                            'padding': '1rem',
                            'borderRadius': '8px',
                            'border': f'2px solid {COLORS["accent"]}',
                            'background': '#16213e',
                            'color': COLORS['text'],
                            'fontFamily': 'VT323, monospace',
                            'fontSize': '18px',
                            'resize': 'vertical'
                        }
                    ),
                    html.Button(
                        "🔮 Analyze Sentiment",
                        id='ai-analyze-btn',
                        className='begin-button',
                        style={'marginTop': '1rem', 'width': '100%'}
                    ),
                ], className="card"),
            ], style={'flex': '2', 'minWidth': '300px'}),
            
            # Result Section
            html.Div([
                html.Div([
                    html.H4("🎯 AI Prediction", style={'color': COLORS['text'], 'marginBottom': '1rem', 'fontFamily': 'VT323, monospace', 'fontSize': '22px', 'textAlign': 'center'}),
                    html.Div(id='ai-result-container', children=[
                        html.Div([
                            html.Span("👈", style={'fontSize': '40px', 'display': 'block', 'marginBottom': '1rem'}),
                            html.P("Enter text and click Analyze to see the AI prediction", 
                                   style={'color': COLORS['muted'], 'textAlign': 'center', 'fontFamily': 'VT323, monospace', 'fontSize': '18px'})
                        ], style={'padding': '2rem', 'textAlign': 'center'})
                    ])
                ], className="card", style={'height': '100%'}),
            ], style={'flex': '1', 'minWidth': '280px'}),
        ], style={'display': 'flex', 'gap': '1.5rem', 'flexWrap': 'wrap'}),
        
        # Model Info
        html.Div([
            html.Div([
                html.H4("ℹ️ About the Model", className="card-title"),
                html.Div([
                    html.Div([
                        html.Span("🤖 Model: ", style={'color': COLORS['muted']}),
                        html.Span("RoBERTa v3.0", style={'color': COLORS['accent'], 'fontWeight': '600'})
                    ], style={'marginBottom': '0.5rem'}),
                    html.Div([
                        html.Span("📊 Accuracy: ", style={'color': COLORS['muted']}),
                        html.Span("96%", style={'color': COLORS['Satisfied'], 'fontWeight': '600'})
                    ], style={'marginBottom': '0.5rem'}),
                    html.Div([
                        html.Span("🎮 Trained on: ", style={'color': COLORS['muted']}),
                        html.Span("Steam game reviews", style={'color': COLORS['text']})
                    ]),
                ], style={'fontFamily': 'VT323, monospace', 'fontSize': '18px'})
            ], className="card"),
        ], style={'marginTop': '1.5rem'}),
    ])

# ============================================
# MAIN LAYOUT
# ============================================

def create_welcome_page():
    """Create an animated welcome/landing page."""
    return html.Div([
        # Animated background particles
        html.Div([
            html.Div(className="particle particle-1"),
            html.Div(className="particle particle-2"),
            html.Div(className="particle particle-3"),
            html.Div(className="particle particle-4"),
            html.Div(className="particle particle-5"),
            html.Div(className="particle particle-6"),
        ], className="particles-container"),
        
        # Main content
        html.Div([
            # Logo/Icon
            html.Div("🎮", className="welcome-icon"),
            
            # Title with typing effect styling
            html.H1("Sentiment Game Analytics", className="welcome-title"),
            
            # Subtitle
            html.P("AI-Powered Steam Reviews Analysis", className="welcome-subtitle"),
            
            # Stats preview
            html.Div([
                html.Div([
                    html.Div(f"{len(df):,}", className="stat-value"),
                    html.Div("Reviews Analyzed", className="stat-label")
                ], className="stat-item"),
                html.Div([
                    html.Div(f"{df['game_name'].nunique()}", className="stat-value"),
                    html.Div("Games", className="stat-label")
                ], className="stat-item"),
                html.Div([
                    html.Div("96%", className="stat-value"),
                    html.Div("Model Accuracy", className="stat-label")
                ], className="stat-item"),
            ], className="welcome-stats"),
            
            # Begin button
            html.Button([
                html.Span("Begin"),
                html.Span("→", className="btn-arrow")
            ], id="begin-btn", className="begin-button"),
            
            # Powered by
            html.P("Powered by RoBERTa Deep Learning", className="welcome-footer")
        ], className="welcome-content")
    ], className="welcome-page", id="welcome-page")


def create_dashboard_page():
    """Create the main dashboard container."""
    return html.Div([
        create_header(),
        
        dcc.Tabs(id='main-tabs', value='tab-1', className='custom-tabs', children=[
            dcc.Tab(label='📊 Dashboard', value='tab-1', className='tab'),
            dcc.Tab(label='🔍 Dev Insights', value='tab-2', className='tab'),
            dcc.Tab(label='⚔️ Game Comparison', value='tab-3', className='tab'),
            dcc.Tab(label='📰 Review Explorer', value='tab-4', className='tab'),
            dcc.Tab(label='🤖 Model Internals', value='tab-5', className='tab'),
            dcc.Tab(label='☁️ Topic Clouds', value='tab-6', className='tab'),
            dcc.Tab(label='🔮 AI Lab', value='tab-7', className='tab'),
        ]),
        
        html.Div(id='tab-content', style={'padding': '1rem 0'})
    ], className="container dashboard-page", id="dashboard-page")


# Loading overlay component
def create_loading_overlay():
    """Create a pixel art loading overlay."""
    return html.Div([
        html.Div("LOADING", className="loading-text"),
        html.Div([
            html.Div(className="loading-dot"),
            html.Div(className="loading-dot"),
            html.Div(className="loading-dot"),
            html.Div(className="loading-dot"),
        ], className="loading-dots"),
        html.Div([
            html.Div(className="loading-bar")
        ], className="loading-bar-container"),
        html.Div("Analyzing game reviews...", className="loading-subtext")
    ], id="loading-overlay", className="loading-overlay hidden")


# Main app layout with page switching
app.layout = html.Div([
    dcc.Store(id='page-state', data='welcome'),
    dcc.Store(id='loading-state', data=False),
    create_loading_overlay(),
    html.Div(id='page-container')
])


# ============================================
# CALLBACKS
# ============================================

# Page switching callback - render welcome or dashboard
@callback(
    Output('page-container', 'children'),
    Input('page-state', 'data')
)
def render_page(page_state):
    if page_state == 'welcome':
        return create_welcome_page()
    else:
        return create_dashboard_page()


# Begin button callback - transition to dashboard
@callback(
    Output('page-state', 'data'),
    Input('begin-btn', 'n_clicks'),
    prevent_initial_call=True
)
def handle_begin_click(n_clicks):
    if n_clicks:
        return 'dashboard'
    return 'welcome'


# Tab content callback
@callback(Output('tab-content', 'children'), Input('main-tabs', 'value'))
def render_tab_content(tab):
    if tab == 'tab-1':
        return create_tab1_dashboard()
    elif tab == 'tab-2':
        return create_tab2_dev_insights()
    elif tab == 'tab-3':
        return create_tab3_comparison()
    elif tab == 'tab-4':
        return create_tab4_explorer()
    elif tab == 'tab-5':
        return create_tab5_model()
    elif tab == 'tab-6':
        return create_tab6_wordcloud()
    elif tab == 'tab-7':
        return create_tab7_live_ai()


# Developer Insights Callbacks
@callback(
    [Output('issue-summary-cards', 'children'),
     Output('priority-matrix', 'figure'),
     Output('recommendations-panel', 'children'),
     Output('issue-trend-chart', 'figure')],
    Input('dev-game-selector', 'value')
)
def update_dev_insights(game):
    game_df = df[df['game_name'] == game]
    issue_counts = extract_issues(game_df)
    total = len(game_df)
    
    # Summary cards
    cards = []
    for category, count in issue_counts.items():
        pct = (count / total * 100) if total > 0 else 0
        cards.append(create_insight_card(category, pct, count, ISSUE_TAXONOMY[category]["color"]))
    
    # Priority Matrix
    matrix_df = calculate_priority_matrix(issue_counts, game_df)
    if not matrix_df.empty and matrix_df['Frequency'].sum() > 0:
        max_freq = max(matrix_df['Frequency'].max(), 10)
        
        # Create color map with clean names
        clean_color_map = {
            "Technical": "#dc2626",
            "Gameplay": "#14b8a6",
            "Content": "#3b82f6",
            "Monetization": "#eab308"
        }
        
        fig_matrix = px.scatter(
            matrix_df, x="Frequency", y="Impact", size="Frequency",
            color="Category", hover_data=["Category"],
            color_discrete_map=clean_color_map,
            size_max=50, template=plotly_template
        )
        fig_matrix.add_hline(y=50, line_dash="dash", line_color="#e2e8f0")
        fig_matrix.add_vline(x=max_freq/2, line_dash="dash", line_color="#e2e8f0")
        fig_matrix.add_annotation(x=max_freq*0.8, y=75, text="CRITICAL", showarrow=False, font=dict(size=10, color="#dc2626"))
        fig_matrix.add_annotation(x=max_freq*0.2, y=75, text="MONITOR", showarrow=False, font=dict(size=10, color="#eab308"))
        fig_matrix.add_annotation(x=max_freq*0.8, y=25, text="BACKLOG", showarrow=False, font=dict(size=10, color="#3b82f6"))
        fig_matrix.add_annotation(x=max_freq*0.2, y=25, text="LOW", showarrow=False, font=dict(size=10, color="#22c55e"))
        fig_matrix.update_layout(
            xaxis_title="Frequency (Mentions)", yaxis_title="Impact (% Negative)", 
            height=460, margin=dict(l=80, r=40, t=70, b=100),
            legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5, title=None),
            paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e'
        )
    else:
        fig_matrix = go.Figure()
        fig_matrix.update_layout(template=plotly_template, height=350)
    
    # Recommendations
    recs = generate_recommendations(issue_counts, total)
    rec_elements = []
    if recs['high']:
        rec_elements.append(html.Div("🔴 High Priority", style={'fontWeight': '600', 'marginBottom': '0.5rem', 'color': '#dc2626'}))
        rec_elements.extend([create_recommendation_card(r, 'priority-high') for r in recs['high']])
    if recs['medium']:
        rec_elements.append(html.Div("🟡 Medium Priority", style={'fontWeight': '600', 'marginBottom': '0.5rem', 'marginTop': '0.75rem', 'color': '#eab308'}))
        rec_elements.extend([create_recommendation_card(r, 'priority-medium') for r in recs['medium']])
    if recs['low']:
        rec_elements.append(html.Div("🟢 Opportunities", style={'fontWeight': '600', 'marginBottom': '0.5rem', 'marginTop': '0.75rem', 'color': '#22c55e'}))
        rec_elements.extend([create_recommendation_card(r, 'priority-low') for r in recs['low']])
    if not rec_elements:
        rec_elements = [html.P("✅ No major issues detected!", style={'color': '#22c55e'})]
    
    # Issue Trend with clean category names
    trend_data = game_df.groupby(['month', 'issue_category']).size().reset_index(name='count')
    # Use clean category colors
    clean_category_colors = {
        "Technical": "#dc2626",
        "Gameplay": "#14b8a6",
        "Content": "#3b82f6",
        "Monetization": "#eab308",
        "Other": "#94a3b8"
    }
    fig_trend = px.line(trend_data, x='month', y='count', color='issue_category',
                        color_discrete_map=clean_category_colors, template=plotly_template)
    fig_trend.update_layout(
        xaxis_title="", yaxis_title="Issue Mentions", height=350,
        margin=dict(l=70, r=40, t=50, b=60),
        xaxis=dict(tickangle=-30),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, title=None)
    )
    
    return cards, fig_matrix, rec_elements, fig_trend


@callback(
    Output('player-quotes', 'children'),
    [Input('dev-game-selector', 'value'),
     Input('quote-category-selector', 'value')]
)
def update_player_quotes(game, category):
    game_df = df[df['game_name'] == game]
    quotes = get_player_quotes(game_df, category)
    if quotes:
        return [create_quote_card(q) for q in quotes]
    return html.P("No reviews found for this category.", style={'color': COLORS['muted']})


# Game Comparison Callbacks
@callback(
    [Output('compare-title-a', 'children'),
     Output('compare-title-b', 'children'),
     Output('compare-pie-a', 'figure'),
     Output('compare-pie-b', 'figure'),
     Output('compare-stats-a', 'children'),
     Output('compare-stats-b', 'children'),
     Output('compare-issues-chart', 'figure'),
     Output('comparison-winner', 'children')],
    [Input('compare-game-a', 'value'),
     Input('compare-game-b', 'value')]
)
def update_comparison(game_a, game_b):
    df_a = df[df['game_name'] == game_a]
    df_b = df[df['game_name'] == game_b]
    
    def make_pie(game_df):
        sat = (game_df['sentiment'] == 2).mean() * 100
        neu = (game_df['sentiment'] == 1).mean() * 100
        dis = (game_df['sentiment'] == 0).mean() * 100
        # Dark text for light colors
        pie_text_colors = ['#1a1a2e', '#1a1a2e', '#ffffff']  # Satisfied, Neutral are light; Dissatisfied is dark
        fig = go.Figure(data=[go.Pie(
            labels=['Satisfied', 'Neutral', 'Dissatisfied'],
            values=[sat, neu, dis],
            hole=0.55,
            marker=dict(colors=[COLORS['Satisfied'], COLORS['Neutral'], COLORS['Dissatisfied']]),
            textinfo='percent+label', textfont=dict(size=14, color=pie_text_colors),
            textposition='inside',
            insidetextorientation='radial'
        )])
        fig.update_layout(template=plotly_template, showlegend=False, height=320, 
                         margin=dict(t=30, b=30, l=30, r=30),
                         uniformtext=dict(minsize=10, mode='hide'),
                         paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e')
        return fig, sat
    
    fig_a, sat_a = make_pie(df_a)
    fig_b, sat_b = make_pie(df_b)
    
    stats_a = html.Div([
        html.Div(f"Satisfaction: {sat_a:.1f}%", style={'fontWeight': '600', 'color': COLORS['Satisfied'], 'fontSize': '18px', 'fontFamily': 'VT323, monospace'}),
        html.Div(f"{len(df_a):,} reviews", style={'fontSize': '16px', 'color': '#ffffff', 'fontFamily': 'VT323, monospace'})
    ])
    stats_b = html.Div([
        html.Div(f"Satisfaction: {sat_b:.1f}%", style={'fontWeight': '600', 'color': COLORS['Satisfied'], 'fontSize': '18px', 'fontFamily': 'VT323, monospace'}),
        html.Div(f"{len(df_b):,} reviews", style={'fontSize': '16px', 'color': '#ffffff', 'fontFamily': 'VT323, monospace'})
    ])
    
    # Issue comparison
    issues_a = extract_issues(df_a)
    issues_b = extract_issues(df_b)
    compare_data = []
    for cat in ISSUE_TAXONOMY.keys():
        pct_a = (issues_a[cat] / len(df_a) * 100) if len(df_a) > 0 else 0
        pct_b = (issues_b[cat] / len(df_b) * 100) if len(df_b) > 0 else 0
        compare_data.append({"Category": cat.split(' ')[1], "Game": game_a, "Issue Rate": pct_a})
        compare_data.append({"Category": cat.split(' ')[1], "Game": game_b, "Issue Rate": pct_b})
    
    compare_df = pd.DataFrame(compare_data)
    fig_issues = px.bar(compare_df, x="Category", y="Issue Rate", color="Game", barmode="group",
                        color_discrete_sequence=["#dc2626", "#3b82f6"], template=plotly_template)
    fig_issues.update_layout(yaxis_title="Issue Rate (%)", xaxis_title="", height=350,
                            margin=dict(t=60, b=60, l=80, r=40),
                            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1),
                            paper_bgcolor='#1a1a2e', plot_bgcolor='#1a1a2e')
    
    # Winner
    winner = game_a if sat_a > sat_b else game_b
    diff = abs(sat_a - sat_b)
    winner_card = html.Div([
        html.Div([
            html.Strong("🏆 Winner: "),
            html.Span(winner, style={'color': 'white'}),
            html.Span(f" (leading by {diff:.1f}%)", style={'opacity': '0.9'})
        ])
    ], className="comparison-winner")
    
    return game_a, game_b, fig_a, fig_b, stats_a, stats_b, fig_issues, winner_card


# Review Explorer Callbacks
@callback(
    [Output('explorer-results', 'children'),
     Output('explorer-count', 'children'),
     Output('explorer-page-info', 'children'),
     Output('explorer-filtered-count', 'data')],
    [Input('explorer-search', 'value'),
     Input('explorer-sentiment', 'value'),
     Input('explorer-game', 'value'),
     Input('explorer-page', 'data')]
)
def update_explorer(search, sentiment, game, page):
    filtered = df.copy()
    
    if search:
        filtered = filtered[filtered['cleaned_text'].str.contains(search, case=False, na=False)]
    if sentiment != 'All':
        sent_map = {'Satisfied': 2, 'Neutral': 1, 'Dissatisfied': 0}
        filtered = filtered[filtered['sentiment'] == sent_map[sentiment]]
    if game != 'All Games':
        filtered = filtered[filtered['game_name'] == game]
    
    total = len(filtered)
    per_page = 8
    total_pages = max(1, (total - 1) // per_page + 1)
    page = min(page, total_pages)
    
    start = (page - 1) * per_page
    end = start + per_page
    page_df = filtered.iloc[start:end]
    
    results = []
    for _, row in page_df.iterrows():
        sent = row.get('sentiment', 1)
        sent_emoji = {0: "🔴", 1: "🟡", 2: "🟢"}.get(sent, "🟡")
        sent_color = {0: COLORS['Dissatisfied'], 1: COLORS['Neutral'], 2: COLORS['Satisfied']}.get(sent, COLORS['Neutral'])
        text = str(row.get('cleaned_text', ''))[:300]
        
        results.append(html.Div([
            html.Div([
                html.Span(f"🎮 {row.get('game_name', 'Unknown')}"),
                html.Span(sent_emoji)
            ], className="review-header"),
            html.Div(text + ('...' if len(str(row.get('cleaned_text', ''))) > 300 else ''), className="review-text")
        ], className="review-card", style={'borderLeft': f'4px solid {sent_color}'}))
    
    count_text = f"Found {total:,} reviews"
    page_info = f"Page {page} of {total_pages}"
    
    return results, count_text, page_info, total


@callback(
    Output('explorer-page', 'data'),
    [Input('explorer-prev', 'n_clicks'),
     Input('explorer-next', 'n_clicks'),
     Input('explorer-search', 'value'),
     Input('explorer-sentiment', 'value'),
     Input('explorer-game', 'value')],
    [State('explorer-page', 'data'),
     State('explorer-filtered-count', 'data')]
)
def update_page(prev_clicks, next_clicks, search, sentiment, game, current_page, total):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 1
    
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger in ['explorer-search', 'explorer-sentiment', 'explorer-game']:
        return 1
    
    per_page = 8
    total_pages = max(1, (total - 1) // per_page + 1)
    
    if trigger == 'explorer-prev':
        return max(1, current_page - 1)
    elif trigger == 'explorer-next':
        return min(total_pages, current_page + 1)
    
    return current_page


# ============================================
# WORDCLOUD TOGGLE BUTTON CALLBACK
# ============================================
@callback(
    [Output('wc-sentiment-filter', 'data'),
     Output('wc-btn-satisfied', 'className'),
     Output('wc-btn-dissatisfied', 'className')],
    [Input('wc-btn-satisfied', 'n_clicks'),
     Input('wc-btn-dissatisfied', 'n_clicks')],
    prevent_initial_call=True
)
def toggle_sentiment_buttons(sat_clicks, dis_clicks):
    """Handle sentiment toggle button clicks"""
    from dash import ctx
    triggered = ctx.triggered_id
    
    if triggered == 'wc-btn-satisfied':
        return 'Satisfied', 'sentiment-toggle-btn active', 'sentiment-toggle-btn'
    elif triggered == 'wc-btn-dissatisfied':
        return 'Dissatisfied', 'sentiment-toggle-btn', 'sentiment-toggle-btn active'
    
    return 'Satisfied', 'sentiment-toggle-btn active', 'sentiment-toggle-btn'


# ============================================
# WORDCLOUD CALLBACK
# ============================================
@callback(
    Output('wordcloud-container', 'children'),
    [Input('wc-sentiment-filter', 'data'),
     Input('wc-game-filter', 'value')]
)
def update_wordcloud(sentiment, game):
    """Generate word cloud based on sentiment and game filter"""
    # Filter data
    filtered_df = df.copy()
    if game != 'All Games':
        filtered_df = filtered_df[filtered_df['game_name'] == game]
    
    # Filter by sentiment
    sentiment_map = {'Satisfied': 2, 'Dissatisfied': 0}
    sent_id = sentiment_map.get(sentiment, 2)
    subset = filtered_df[filtered_df['sentiment'] == sent_id]
    
    if subset.empty:
        return html.Div([
            html.Span("😕", style={'fontSize': '60px', 'display': 'block', 'marginBottom': '1rem'}),
            html.P(f"No {sentiment.lower()} reviews found for this selection", 
                   style={'color': COLORS['muted'], 'fontFamily': 'VT323, monospace', 'fontSize': '18px'})
        ], style={'textAlign': 'center', 'padding': '2rem'})
    
    # Generate word cloud
    text_corpus = " ".join(subset['cleaned_text'].astype(str).tolist())
    # Remove common words and patterns
    text_corpus = re.sub(r'\bd\d+\b', '', text_corpus)
    
    # Choose colormap based on sentiment
    cloud_colormap = 'viridis' if sentiment == 'Satisfied' else 'magma'
    
    try:
        # Use WordCloud's built-in STOPWORDS plus custom gaming-specific words
        from wordcloud import STOPWORDS
        custom_stopwords = STOPWORDS.union({
            # Gaming-generic terms
            'game', 'games', 'play', 'playing', 'played', 'player', 'players',
            'steam', 'review', 'reviews', 'buy', 'bought', 'purchase',
            # Common filler words
            'get', 'got', 'getting', 'one', 'like', 'really', 'would', 'could', 
            'also', 'much', 'even', 'still', 'make', 'makes', 'made', 'well', 
            'way', 'thing', 'things', 'dont', 'didnt', 'doesnt', 'wont', 'cant',
            'want', 'wanted', 'think', 'know', 'feel', 'feels', 'go', 'going',
            'come', 'coming', 'take', 'took', 'give', 'gave', 'see', 'saw',
            'thats', 'theres', 'ive', 'youre', 'theyre', 'weve', 'youve',
            'im', 'hes', 'shes', 'its', 'isnt', 'arent', 'wasnt', 'werent',
            'ever', 'never', 'always', 'sometimes', 'maybe', 'probably',
            'actually', 'basically', 'definitely', 'especially', 'literally',
            'pretty', 'quite', 'rather', 'somewhat', 'almost', 'enough',
            'lot', 'lots', 'many', 'few', 'bit', 'little', 'big', 'small',
            'first', 'last', 'next', 'new', 'old', 'long', 'short',
            'said', 'says', 'say', 'tell', 'told', 'ask', 'asked',
            'put', 'use', 'used', 'using', 'try', 'tried', 'trying',
            'need', 'needs', 'needed', 'keep', 'keeps', 'kept',
            'find', 'found', 'look', 'looking', 'looks', 'looked',
            've', 're', 'll', 'd', 's', 't', 'u', 'ur', 'r',
            'yes', 'no', 'ok', 'okay', 'sure', 'yeah', 'yep', 'nope',
            'hour', 'hours', 'minute', 'minutes', 'time', 'times', 'day', 'days',
            'worth', 'money', 'dollar', 'dollars', 'price', 'free', 'paid'
        })
        
        wc = WordCloud(
            width=800, 
            height=500, 
            background_color='white',
            mode='RGBA',
            colormap=cloud_colormap,
            collocations=False,
            stopwords=custom_stopwords,
            max_words=100,
            min_font_size=10
        ).generate(text_corpus)
        
        # Convert to image
        fig, ax = plt.subplots(figsize=(10, 6.25), facecolor='white')
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_facecolor('white')
        plt.tight_layout(pad=0)
        
        # Save to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor='white', edgecolor='none', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        plt.close(fig)
        
        # Encode to base64
        img_str = base64.b64encode(buf.read()).decode()
        
        return html.Img(
            src=f'data:image/png;base64,{img_str}',
            style={'width': '100%', 'borderRadius': '8px'}
        )
    except Exception as e:
        return html.Div([
            html.Span("⚠️", style={'fontSize': '40px', 'display': 'block', 'marginBottom': '1rem'}),
            html.P(f"Error generating word cloud: {str(e)}", 
                   style={'color': COLORS['Dissatisfied'], 'fontFamily': 'VT323, monospace', 'fontSize': '16px'})
        ], style={'textAlign': 'center', 'padding': '2rem'})


# ============================================
# LIVE AI LAB CALLBACK
# ============================================
@callback(
    Output('ai-result-container', 'children'),
    Input('ai-analyze-btn', 'n_clicks'),
    State('ai-text-input', 'value'),
    prevent_initial_call=True
)
def analyze_sentiment(n_clicks, text):
    """Analyze sentiment of user-provided text"""
    if not text or not text.strip():
        return html.Div([
            html.Span("✍️", style={'fontSize': '40px', 'display': 'block', 'marginBottom': '1rem'}),
            html.P("Please enter some text to analyze", 
                   style={'color': COLORS['muted'], 'fontFamily': 'VT323, monospace', 'fontSize': '18px'})
        ], style={'textAlign': 'center', 'padding': '2rem'})
    
    # Get prediction
    label, confidence, error = predict_sentiment(text)
    
    if error:
        return html.Div([
            html.Span("⚠️", style={'fontSize': '40px', 'display': 'block', 'marginBottom': '1rem'}),
            html.P(f"Error: {error}", 
                   style={'color': COLORS['Dissatisfied'], 'fontFamily': 'VT323, monospace', 'fontSize': '16px'})
        ], style={'textAlign': 'center', 'padding': '2rem'})
    
    # Emoji and color mapping
    emoji_map = {'Satisfied': '🟢', 'Neutral': '🟡', 'Dissatisfied': '🔴'}
    color_map = {'Satisfied': COLORS['Satisfied'], 'Neutral': COLORS['Neutral'], 'Dissatisfied': COLORS['Dissatisfied']}
    
    return html.Div([
        html.Div([
            html.Span(emoji_map.get(label, '🔵'), style={'fontSize': '60px', 'display': 'block', 'marginBottom': '0.5rem'}),
            html.H2(label, style={'color': color_map.get(label, COLORS['text']), 'fontFamily': 'VT323, monospace', 'fontSize': '32px', 'marginBottom': '1rem'}),
        ]),
        html.Div([
            html.Label("Confidence Score:", style={'color': COLORS['muted'], 'fontFamily': 'VT323, monospace', 'fontSize': '16px', 'marginBottom': '0.5rem', 'display': 'block'}),
            html.Div([
                html.Div(style={
                    'width': f'{confidence * 100}%',
                    'height': '12px',
                    'backgroundColor': color_map.get(label, COLORS['accent']),
                    'borderRadius': '6px',
                    'transition': 'width 0.5s ease'
                })
            ], style={
                'width': '100%',
                'height': '12px',
                'backgroundColor': '#16213e',
                'borderRadius': '6px',
                'overflow': 'hidden',
                'marginBottom': '0.5rem'
            }),
            html.Span(f"{confidence:.1%}", style={'color': color_map.get(label, COLORS['text']), 'fontFamily': 'VT323, monospace', 'fontSize': '20px', 'fontWeight': '600'})
        ], style={'marginTop': '1rem'})
    ], style={'textAlign': 'center', 'padding': '1.5rem'})


# ============================================
# RUN SERVER
# ============================================

# Export server for gunicorn (required for production deployment)
server = app.server

if __name__ == '__main__':
    import os
    # Check if running in production (HuggingFace Spaces)
    is_production = os.environ.get('SPACE_ID') is not None
    
    if is_production:
        # Production mode for HuggingFace Spaces
        app.run(host='0.0.0.0', port=7860, debug=False)
    else:
        # Development mode
        print("\n" + "="*50)
        print("🎮 Sentiment Game Analytics Dashboard")
        print("="*50)
        print("Starting server...")
        print("Open http://127.0.0.1:8050 in your browser")
        print("="*50 + "\n")
        app.run(debug=True)
