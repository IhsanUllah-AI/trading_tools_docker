import json
import os
from datetime import datetime, timezone
from flask import Flask, render_template, request, session, jsonify
from flask_apscheduler import APScheduler

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

scheduler = APScheduler()
scheduler.init_app(app)

# Import analysis functions
from fibonacci import run_fibonacci_analysis
from elliott import run_elliott_analysis
from ichimoku import run_ichimoku_analysis
from wyckoff import run_wyckoff_analysis
from gann import run_gann_analysis

# Data directory for JSON storage
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

# JSON file paths
ACTIVE_TRADES_FILE = os.path.join(DATA_DIR, 'active_trades.json')
TRADE_HISTORY_FILE = os.path.join(DATA_DIR, 'trade_history.json')
AUTO_REFRESH_FILE = os.path.join(DATA_DIR, 'auto_refresh_state.json')
COMBINED_SETTINGS_FILE = os.path.join(DATA_DIR, 'combined_settings.json')
ANALYSIS_CONFIG_FILE = os.path.join(DATA_DIR, 'analysis_config.json')
RUNNING_TIME_FILE = os.path.join(DATA_DIR, 'running_time.json')
BUDGET_FILE = os.path.join(DATA_DIR, 'budget.json')

# Add after imports
FIXED_TRADE_AMOUNT = 500.0  # Fixed $500 per trade
MAKER_FEE = 0.001  # 0.1% maker fee
TAKER_FEE = 0.001  # 0.1% taker fee

def calculate_trade_costs(investment_amount, is_opening=True):
    """Calculate fees and net investment amount"""
    fee_rate = TAKER_FEE if is_opening else MAKER_FEE
    fee_amount = investment_amount * fee_rate
    net_amount = investment_amount - fee_amount
    return fee_amount, net_amount

def load_budget():
    """Load budget data from file"""
    default_budget = {
        'total_budget': 5000.0,
        'used_budget': 0.0,
        'remaining_budget': 5000.0,
        'initial_budget': 5000.0,
        'total_fees': 0.0,
        'total_invested': 0.0
    }
    loaded_budget = load_json_data(BUDGET_FILE, default_budget)
    
    # Ensure all fields are present (backward compatibility)
    for key in default_budget.keys():
        if key not in loaded_budget:
            loaded_budget[key] = default_budget[key]
    
    return loaded_budget

def update_budget(investment_amount, fee_amount, action="use", gross_profit_usd=0.0):
    """Update budget when trade is opened or closed"""
    budget = load_budget()
    
    if action == "use":
        budget['used_budget'] += investment_amount + fee_amount
        budget['remaining_budget'] = budget['total_budget'] - budget['used_budget']
        budget['total_fees'] += fee_amount
        budget['total_invested'] += investment_amount
    elif action == "return":  # When trade is closed
        budget['used_budget'] -= investment_amount
        budget['total_fees'] += fee_amount
        budget['used_budget'] += fee_amount
        budget['total_invested'] -= investment_amount
        budget['total_budget'] += gross_profit_usd
        budget['remaining_budget'] = budget['total_budget'] - budget['used_budget']
    
    save_budget(budget)
    return True

def save_budget(budget_data):
    """Save budget data to file"""
    return save_json_data(BUDGET_FILE, budget_data)

def can_open_trade():
    """Check if there's enough budget to open a new $500 trade"""
    budget = load_budget()
    total_cost = FIXED_TRADE_AMOUNT + (FIXED_TRADE_AMOUNT * TAKER_FEE)
    return budget['remaining_budget'] >= total_cost, total_cost, FIXED_TRADE_AMOUNT

def load_json_data(file_path, default_data):
    """Load JSON data from file, return default if file doesn't exist"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return default_data

def save_json_data(file_path, data):
    """Save data to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving {file_path}: {e}")
        return False

def load_auto_refresh_state():
    """Load auto-refresh state from file"""
    default_state = {'enabled': False, 'last_analysis_time': None}
    return load_json_data(AUTO_REFRESH_FILE, default_state)

def save_auto_refresh_state(state):
    """Save auto-refresh state to file"""
    return save_json_data(AUTO_REFRESH_FILE, state)

def load_combined_settings():
    """Load combined analysis settings from file"""
    default_settings = {
        'confidence_threshold': 0.6,
        'min_tool_agreement': 2,
        'risk_reward_ratio': '1:2',
        'tool_weights': {
            'fibonacci': 1.0,
            'elliott': 1.0,
            'ichimoku': 1.0,
            'wyckoff': 1.0,
            'gann': 0.5
        }
    }
    return load_json_data(COMBINED_SETTINGS_FILE, default_settings)

def save_combined_settings(settings):
    """Save combined analysis settings to file"""
    return save_json_data(COMBINED_SETTINGS_FILE, settings)

def load_trade_data():
    """Load trade data from JSON files"""
    active_trades_default = {'fibonacci': {}, 'elliott': {}, 'ichimoku': {}, 'wyckoff': {}, 'gann': {}, 'combined': {}}
    trade_history_default = {'fibonacci': [], 'elliott': [], 'ichimoku': [], 'wyckoff': [], 'gann': [], 'combined': []}
    
    active_trades = load_json_data(ACTIVE_TRADES_FILE, active_trades_default)
    trade_history = load_json_data(TRADE_HISTORY_FILE, trade_history_default)
    
    return active_trades, trade_history

def save_trade_data(active_trades, trade_history):
    """Save trade data to JSON files"""
    success1 = save_json_data(ACTIVE_TRADES_FILE, active_trades)
    success2 = save_json_data(TRADE_HISTORY_FILE, trade_history)
    return success1 and success2

def load_analysis_config():
    """Load persistent analysis config"""
    default_config = {
        'selected_tools': ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann'],
        'symbols': ['BTCUSDT'],
        'interval': '5m',
        'candle_limit': 1000,
        'fib_window': 60,
        'fib_threshold': 0.003,
        'elliott_thresholds': {'Minor': 0.005, 'Intermediate': 0.015, 'Major': 0.03},
        'elliott_degrees': ['Minor', 'Intermediate', 'Major'],
        'use_smoothing': False,
        'smooth_period': 3,
        'show_ema': False,
        'show_bb': False,
        'show_volume': False,
        'show_rsi': False,
        'show_macd': False,
        'show_ema_ichimoku': False,
        'show_bb_ichimoku': False,
        'show_volume_ichimoku': False,
        'show_rsi_ichimoku': False,
        'show_macd_ichimoku': False,
        'show_ema_wyckoff': False,
        'show_bb_wyckoff': False,
        'show_volume_wyckoff': False,
        'show_rsi_wyckoff': False,
        'show_macd_wyckoff': False,
        'gann_subtools': ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'],
        'pivot_choice': 'Auto (based on trend)',
        'show_ema_gann': False,
        'show_bb_gann': False,
        'show_volume_gann': False,
        'show_rsi_gann': False,
        'show_macd_gann': False,
        'enable_buy': True,  # New: Enable buy signals
        'enable_sell': True  # New: Enable sell signals
    }
    return load_json_data(ANALYSIS_CONFIG_FILE, default_config)

def save_analysis_config(config):
    """Save persistent analysis config"""
    return save_json_data(ANALYSIS_CONFIG_FILE, config)

def load_running_time():
    """Load running start time"""
    default = {'start_time': None}
    return load_json_data(RUNNING_TIME_FILE, default)

def save_running_time(data):
    """Save running start time"""
    return save_json_data(RUNNING_TIME_FILE, data)

# Initialize session data
def init_session():
    # Load trade data from JSON files
    active_trades, trade_history = load_trade_data()
    
    # Initialize session with loaded data
    if 'active_trades' not in session:
        session['active_trades'] = active_trades
    if 'trade_history' not in session:
        session['trade_history'] = trade_history
    
    # Load auto-refresh state
    auto_refresh_state = load_auto_refresh_state()
    if 'auto_refresh_enabled' not in session:
        session['auto_refresh_enabled'] = auto_refresh_state.get('enabled', False)
    if 'last_analysis_time' not in session:
        session['last_analysis_time'] = auto_refresh_state.get('last_analysis_time')
    
    # Load combined settings
    combined_settings = load_combined_settings()
    if 'combined_settings' not in session:
        session['combined_settings'] = combined_settings
    
    # Load and initialize budget
    budget = load_budget()
    session['budget'] = budget

# Normalize analysis output to ensure required keys
def normalize_analysis(analysis, tool):
    if not analysis:
        return None
    default_keys = {
        'fibonacci': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'trend': 'none',
            'confidence': 'low',
            'signals': [],
            'signal_descriptions': [],
            'trade_action': None,
            'entry_price': None,
            'stop_loss': None,
            'take_profit': None,
            'risk_reward_ratio': None,
            'swing_high': 0.0,
            'swing_low': 0.0,
            'trend_class': 'trend-none',
            'confidence_class': 'confidence-low',
            'last_candle': 'N/A',
            'data_from': 'N/A',
            'data_to': 'N/A',
            'chart_html': '',
            'fib_html': '',
            'closest_info': 'No levels detected',
            'fib_explanation': '',
            'rsi_value': 0.0,
            'ema_value': 0.0,
            'ema_rel': 'N/A',
            'atr_value': 0.0,
            'bb_value': 0.0,
            'bb_status': 'N/A'
        },
        'elliott': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'wave_data_by_degree': {},
            'charts': {},
            'technical_indicators': {
                'rsi': 0.0,
                'ema': 0.0,
                'bb_percent': 0.0
            }
        },
        'ichimoku': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'trend': 'none',
            'confidence': 'low',
            'cloud_bullish': False,
            'signals': [],
            'reasons_not_met': [],
            'chart_html': '',
            'technical_indicators': {
                'rsi': 0.0,
                'ema': 0.0,
                'atr': 0.0,
                'bb_percent': 0.0,
                'tenkan_sen': 0.0,
                'kijun_sen': 0.0,
                'senkou_span_a': 0.0,
                'senkou_span_b': 0.0
            }
        },
        'wyckoff': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'phase': 'none',
            'confidence': 'low',
            'sideways_count': 0,
            'signals': [],
            'reasons_not_met': [],
            'chart_html': '',
            'technical_indicators': {
                'rsi': 0.0,
                'ema_short': 0.0,
                'ema_long': 0.0,
                'atr': 0.0,
                'bb_percent': 0.0,
                'support': 0.0,
                'resistance': 0.0
            }
        },
        'gann': {
            'symbol': 'N/A',
            'interval': 'N/A',
            'current_price': 0.0,
            'trend': 'none',
            'confidence': 'low',
            'pivot_type': 'N/A',
            'price_position': 'N/A',
            'nearest_support': 0.0,
            'nearest_resistance': 0.0,
            'subtools': ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'],
            'charts': {},
            'technical_indicators': {
                'rsi': 0.0,
                'ema': 0.0,
                'atr': 0.0,
                'bb_percent': 0.0
            }
        }
    }
    normalized = default_keys[tool].copy()
    if analysis:
        normalized.update(analysis)
    return normalized

def run_analysis_for_tool(tool, symbols, interval, candle_limit, config):
    """Run analysis for a specific tool and return results. Uses config dict instead of request_form."""
    analyses = {}
    
    if tool == 'fibonacci':
        window = config.get('fib_window', 60)
        fib_threshold = config.get('fib_threshold', 0.003)
        
        for symbol in symbols:
            try:
                analysis = run_fibonacci_analysis(symbol, interval, candle_limit, window, fib_threshold)
                if analysis:
                    analysis = normalize_analysis(analysis, 'fibonacci')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Fibonacci analysis for {symbol}: {e}")
                
    elif tool == 'elliott':
        thresholds = config.get('elliott_thresholds', {'Minor': 0.005, 'Intermediate': 0.015, 'Major': 0.03})
        
        selected_degrees = config.get('elliott_degrees', ['Minor', 'Intermediate', 'Major'])
        use_smoothing = config.get('use_smoothing', False)
        smooth_period = config.get('smooth_period', 3)
        
        show_ema = config.get('show_ema', False)
        show_bb = config.get('show_bb', False)
        show_volume = config.get('show_volume', False)
        show_rsi = config.get('show_rsi', False)
        show_macd = config.get('show_macd', False)
        
        for symbol in symbols:
            try:
                analysis = run_elliott_analysis(
                    symbol, interval, candle_limit, thresholds, selected_degrees,
                    use_smoothing, smooth_period, show_ema, show_bb, show_volume, show_rsi, show_macd
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'elliott')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Elliott analysis for {symbol}: {e}")
                
    elif tool == 'ichimoku':
        show_ema_ichimoku = config.get('show_ema_ichimoku', False)
        show_bb_ichimoku = config.get('show_bb_ichimoku', False)
        show_volume_ichimoku = config.get('show_volume_ichimoku', False)
        show_rsi_ichimoku = config.get('show_rsi_ichimoku', False)
        show_macd_ichimoku = config.get('show_macd_ichimoku', False)
        
        for symbol in symbols:
            try:
                analysis = run_ichimoku_analysis(
                    symbol, interval, candle_limit,
                    show_ema_ichimoku, show_bb_ichimoku, show_volume_ichimoku, 
                    show_rsi_ichimoku, show_macd_ichimoku
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'ichimoku')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Ichimoku analysis for {symbol}: {e}")
                
    elif tool == 'wyckoff':
        show_ema_wyckoff = config.get('show_ema_wyckoff', False)
        show_bb_wyckoff = config.get('show_bb_wyckoff', False)
        show_volume_wyckoff = config.get('show_volume_wyckoff', False)
        show_rsi_wyckoff = config.get('show_rsi_wyckoff', False)
        show_macd_wyckoff = config.get('show_macd_wyckoff', False)
        
        for symbol in symbols:
            try:
                analysis = run_wyckoff_analysis(
                    symbol, interval, candle_limit,
                    show_ema_wyckoff, show_bb_wyckoff, show_volume_wyckoff,
                    show_rsi_wyckoff, show_macd_wyckoff
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'wyckoff')
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Wyckoff analysis for {symbol}: {e}")
                
    elif tool == 'gann':
        gann_subtools = config.get('gann_subtools', ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'])
        
        pivot_choice = config.get('pivot_choice', 'Auto (based on trend)')
        show_ema_gann = config.get('show_ema_gann', False)
        show_bb_gann = config.get('show_bb_gann', False)
        show_volume_gann = config.get('show_volume_gann', False)
        show_rsi_gann = config.get('show_rsi_gann', False)
        show_macd_gann = config.get('show_macd_gann', False)
        
        for symbol in symbols:
            try:
                analysis = run_gann_analysis(
                    symbol, interval, candle_limit,
                    gann_subtools, pivot_choice,
                    show_ema_gann, show_bb_gann, show_volume_gann, show_rsi_gann, show_macd_gann
                )
                if analysis:
                    analysis = normalize_analysis(analysis, 'gann')
                    if not analysis.get('subtools'):
                        analysis['subtools'] = gann_subtools
                    analyses[symbol] = analysis
            except Exception as e:
                print(f"Error running Gann analysis for {symbol}: {e}")
    
    return analyses

def manage_trades(tool, analyses, session_data, interval, enable_buy=True, enable_sell=True):
    """Manage active trades and trade history for a specific tool with fixed $500 trade size"""
    active_trades = session_data['active_trades']
    trade_history = session_data['trade_history']
    budget = load_budget()
    
    for symbol, analysis in analyses.items():
        current_price = analysis['current_price']
        
        if tool == 'elliott':
            for degree in analysis.get('wave_data_by_degree', {}).keys():
                trade_key = f"{symbol}_{degree}"
                wave_data = analysis['wave_data_by_degree'].get(degree, {})
                
                # Check if existing trade should be closed
                if trade_key in active_trades[tool]:
                    trade = active_trades[tool][trade_key]
                    hit_sl = (trade['action'] == "BUY" and current_price <= trade['stop_loss']) or \
                             (trade['action'] == "SELL" and current_price >= trade['stop_loss'])
                    hit_tp = (trade['action'] == "BUY" and current_price >= trade['take_profit']) or \
                             (trade['action'] == "SELL" and current_price <= trade['take_profit'])
                    
                    if hit_sl or hit_tp:
                        # Calculate closing fee and net proceeds
                        closing_fee, net_proceeds = calculate_trade_costs(trade['invested_amount'], is_opening=False)
                        
                        # Calculate profit/loss including fees
                        if trade['action'] == "BUY":
                            gross_profit = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                            gross_profit_usd = (current_price - trade['entry_price']) * (trade['net_investment'] / trade['entry_price'])
                            net_profit_usd = gross_profit_usd - trade['entry_fee'] - closing_fee
                        else:  # SELL
                            gross_profit = (trade['entry_price'] - current_price) / trade['entry_price'] * 100
                            gross_profit_usd = (trade['entry_price'] - current_price) * (trade['net_investment'] / trade['entry_price'])
                            net_profit_usd = gross_profit_usd - trade['entry_fee'] - closing_fee
                        
                        net_profit_percent = (net_profit_usd / trade['invested_amount']) * 100
                        outcome = 'win' if net_profit_usd > 0 else 'loss'
                        
                        closed_trade = trade.copy()
                        closed_trade.update({
                            'outcome': outcome,
                            'close_price': current_price,
                            'profit_pct': gross_profit,  # Gross profit percentage
                            'net_profit_pct': net_profit_percent,  # Net profit percentage after fees
                            'net_profit_usd': net_profit_usd,  # Net profit in USD
                            'gross_profit_usd': gross_profit_usd,
                            'closing_fee': closing_fee,
                            'total_fees': trade['entry_fee'] + closing_fee,
                            'close_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                            'interval': interval
                        })
                        trade_history[tool].append(closed_trade)
                        
                        # Return budget when trade closes (only the invested amount, fees are already deducted)
                        update_budget(trade['invested_amount'], closing_fee, "return", gross_profit_usd=gross_profit_usd)
                        
                        del active_trades[tool][trade_key]
                        print(f"{tool.capitalize()} trade closed for {symbol} ({degree}): {outcome}, Net Profit: ${net_profit_usd:.2f}")

                # Check if new trade should be opened
                if trade_key not in active_trades[tool] and wave_data.get('signals'):
                    for signal in wave_data['signals']:
                        action = signal['type']
                        if (action == 'BUY' and not enable_buy) or (action == 'SELL' and not enable_sell):
                            print(f"Skipping {action} signal for {symbol} ({degree}) as it is disabled.")
                            continue
                        can_trade, total_cost, investment_amount = can_open_trade()
                        
                        if can_trade:
                            entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                            
                            active_trade = {
                                'symbol': symbol,
                                'degree': degree,
                                'action': action,
                                'entry_price': signal['entry_price'],
                                'stop_loss': signal['sl'],
                                'take_profit': signal['tp'],
                                'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                                'reason': signal['reason'],
                                'interval': interval,
                                'position_size': 100 * (investment_amount / budget['total_budget']),  # Calculate percentage
                                'invested_amount': investment_amount,
                                'entry_fee': entry_fee,
                                'net_investment': net_investment
                            }
                            active_trades[tool][trade_key] = active_trade
                            
                            # Use budget (investment amount + fee)
                            update_budget(investment_amount, entry_fee, "use")
                            print(f"Elliott trade opened for {symbol} ({degree}): {action}, Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                        else:
                            print(f"Insufficient budget for Elliott trade on {symbol} ({degree}). Required: ${total_cost:.2f}, Available: ${budget['remaining_budget']:.2f}")
                        break
        else:
            trade_key = symbol
        
            # Check if existing trade should be closed
            if trade_key in active_trades[tool]:
                trade = active_trades[tool][trade_key]
                hit_sl = (trade['action'] == "BUY" and current_price <= trade['stop_loss']) or \
                         (trade['action'] == "SELL" and current_price >= trade['stop_loss'])
                hit_tp = (trade['action'] == "BUY" and current_price >= trade['take_profit']) or \
                         (trade['action'] == "SELL" and current_price <= trade['take_profit'])
                
                if hit_sl or hit_tp:
                    # Calculate closing fee and net proceeds
                    closing_fee, net_proceeds = calculate_trade_costs(trade['invested_amount'], is_opening=False)
                    
                    # Calculate profit/loss including fees
                    if trade['action'] == "BUY":
                        gross_profit = (current_price - trade['entry_price']) / trade['entry_price'] * 100
                        gross_profit_usd = (current_price - trade['entry_price']) * (trade['net_investment'] / trade['entry_price'])
                        net_profit_usd = gross_profit_usd - trade['entry_fee'] - closing_fee
                    else:  # SELL
                        gross_profit = (trade['entry_price'] - current_price) / trade['entry_price'] * 100
                        gross_profit_usd = (trade['entry_price'] - current_price) * (trade['net_investment'] / trade['entry_price'])
                        net_profit_usd = gross_profit_usd - trade['entry_fee'] - closing_fee
                    
                    net_profit_percent = (net_profit_usd / trade['invested_amount']) * 100
                    outcome = 'win' if net_profit_usd > 0 else 'loss'
                    
                    closed_trade = trade.copy()
                    closed_trade.update({
                        'outcome': outcome,
                        'close_price': current_price,
                        'profit_pct': gross_profit,  # Gross profit percentage
                        'net_profit_pct': net_profit_percent,  # Net profit percentage after fees
                        'net_profit_usd': net_profit_usd,  # Net profit in USD
                        'gross_profit_usd': gross_profit_usd,
                        'closing_fee': closing_fee,
                        'total_fees': trade['entry_fee'] + closing_fee,
                        'close_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'interval': interval
                    })
                    trade_history[tool].append(closed_trade)
                    
                    # Return budget when trade closes
                    update_budget(trade['invested_amount'], closing_fee, "return", gross_profit_usd=gross_profit_usd)
                    
                    del active_trades[tool][trade_key]
                    print(f"{tool.capitalize()} trade closed for {symbol}: {outcome}, Net Profit: ${net_profit_usd:.2f}")

            # Check if new trade should be opened - ONLY FOR BUY/SELL ACTIONS
            if trade_key not in active_trades[tool]:
                trade_action = analysis.get('trade_action') or analysis.get('action')
                
                # For combined analysis, only create trades for BUY/SELL, not HOLD
                if tool == 'combined':
                    if trade_action in ['BUY', 'SELL']:
                        if (trade_action == 'BUY' and not enable_buy) or (trade_action == 'SELL' and not enable_sell):
                            print(f"Skipping {trade_action} signal for {symbol} (combined) as it is disabled.")
                            continue
                        can_trade, total_cost, investment_amount = can_open_trade()
                        
                        if can_trade:
                            entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                            
                            active_trade = {
                                'symbol': symbol,
                                'action': trade_action,
                                'entry_price': analysis.get('entry_price', current_price),
                                'stop_loss': analysis.get('stop_loss'),
                                'take_profit': analysis.get('take_profit'),
                                'position_size': 100 * (investment_amount / budget['total_budget']),
                                'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                                'confidence': analysis.get('confidence'),
                                'agreement_level': analysis.get('agreement_level'),
                                'interval': interval,
                                'invested_amount': investment_amount,
                                'entry_fee': entry_fee,
                                'net_investment': net_investment
                            }
                            active_trades[tool][trade_key] = active_trade
                            
                            # Use budget
                            update_budget(investment_amount, entry_fee, "use")
                            print(f"Combined trade opened for {symbol}: {trade_action}, Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                        else:
                            print(f"Insufficient budget for combined trade on {symbol}. Required: ${total_cost:.2f}, Available: ${budget['remaining_budget']:.2f}")
                else:
                    # For other tools, use fixed $500 trade size
                    if trade_action in ['BUY', 'SELL']:
                        if (trade_action == 'BUY' and not enable_buy) or (trade_action == 'SELL' and not enable_sell):
                            print(f"Skipping {trade_action} signal for {symbol} ({tool}) as it is disabled.")
                            continue
                        can_trade, total_cost, investment_amount = can_open_trade()
                        
                        if can_trade:
                            entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                            
                            active_trade = {
                                'symbol': symbol,
                                'action': trade_action,
                                'entry_price': analysis.get('entry_price', current_price),
                                'stop_loss': analysis.get('stop_loss'),
                                'take_profit': analysis.get('take_profit'),
                                'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                                'signals': analysis.get('signals', []),
                                'signal_descriptions': analysis.get('signal_descriptions', []),
                                'confidence': analysis.get('confidence'),
                                'interval': interval,
                                'position_size': 100 * (investment_amount / budget['total_budget']),
                                'invested_amount': investment_amount,
                                'entry_fee': entry_fee,
                                'net_investment': net_investment
                            }
                            active_trades[tool][trade_key] = active_trade
                            
                            # Use budget
                            update_budget(investment_amount, entry_fee, "use")
                            print(f"{tool.capitalize()} trade opened for {symbol}: {trade_action}, Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                        else:
                            print(f"Insufficient budget for {tool} trade on {symbol}. Required: ${total_cost:.2f}, Available: ${budget['remaining_budget']:.2f}")
                    elif analysis.get('signals'):
                        for signal in analysis['signals']:
                            if not isinstance(signal, dict):
                                continue
                            
                            action = signal.get('type') or signal.get('action')
                            if not action or action not in ['BUY', 'SELL']:
                                continue
                            if (action == 'BUY' and not enable_buy) or (action == 'SELL' and not enable_sell):
                                print(f"Skipping {action} signal for {symbol} ({tool}) as it is disabled.")
                                continue
                                
                            can_trade, total_cost, investment_amount = can_open_trade()
                            
                            if can_trade:
                                entry_fee, net_investment = calculate_trade_costs(investment_amount, is_opening=True)
                                
                                active_trade = {
                                    'symbol': symbol,
                                    'action': action,
                                    'entry_price': signal.get('entry_price', current_price),
                                    'stop_loss': signal.get('sl', signal.get('stop_loss')),
                                    'take_profit': signal.get('tp', signal.get('take_profit')),
                                    'entry_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                                    'reason': signal.get('reason', 'No reason provided'),
                                    'interval': interval,
                                    'position_size': 100 * (investment_amount / budget['total_budget']),
                                    'invested_amount': investment_amount,
                                    'entry_fee': entry_fee,
                                    'net_investment': net_investment
                                }
                                active_trades[tool][trade_key] = active_trade
                                
                                # Use budget
                                update_budget(investment_amount, entry_fee, "use")
                                print(f"{tool.capitalize()} trade opened for {symbol}: {action}, Investment: ${investment_amount:.2f}, Fee: ${entry_fee:.2f}")
                                break
                            else:
                                print(f"Insufficient budget for {tool} trade on {symbol}. Required: ${total_cost:.2f}, Available: ${budget['remaining_budget']:.2f}")
                                break
# Helper functions for combined analysis
def convert_confidence_to_numeric(confidence_str):
    """Convert confidence string to numeric value"""
    confidence_map = {
        'very high': 0.9,
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4,
        'very low': 0.2
    }
    return confidence_map.get(confidence_str.lower(), 0.5)

def convert_numeric_to_confidence(confidence_numeric):
    """Convert numeric confidence to string"""
    if confidence_numeric >= 0.8:
        return 'High'
    elif confidence_numeric >= 0.6:
        return 'Medium'
    elif confidence_numeric >= 0.4:
        return 'Low'
    else:
        return 'Very Low'

def analyze_gann_information(gann_data, current_price):
    """Analyze Gann data to extract trading bias"""
    if not gann_data:
        return {'bias': 'neutral', 'confidence': 'low', 'reasons': ['No Gann data available']}
    
    reasons = []
    bullish_signals = 0
    bearish_signals = 0
    
    # Analyze Gann levels
    if gann_data.get('price_position') == 'Near Support':
        bullish_signals += 1
        reasons.append("Price near Gann support level")
    
    if gann_data.get('price_position') == 'Near Resistance':
        bearish_signals += 1
        reasons.append("Price near Gann resistance level")
    
    # Analyze Gann tools
    analysis_details = gann_data.get('gann_analysis_details', {})
    for tool_name, tool_data in analysis_details.items():
        signal = tool_data.get('signal', '').lower()
        if 'bullish' in signal:
            bullish_signals += 1
            reasons.append(f"{tool_name}: Bullish signal")
        elif 'bearish' in signal:
            bearish_signals += 1
            reasons.append(f"{tool_name}: Bearish signal")
    
    # Determine overall bias
    if bullish_signals > bearish_signals:
        bias = 'bullish'
        confidence = 'medium' if (bullish_signals - bearish_signals) >= 2 else 'low'
    elif bearish_signals > bullish_signals:
        bias = 'bearish'
        confidence = 'medium' if (bearish_signals - bullish_signals) >= 2 else 'low'
    else:
        bias = 'neutral'
        confidence = 'low'
        reasons.append("Mixed signals from Gann tools")
    
    return {
        'bias': bias,
        'confidence': confidence,
        'reasons': reasons,
        'detailed_analysis': analysis_details
    }

def calculate_tp_sl_rr(current_price, action, atr, rr_ratio):
    """Calculate take profit and stop loss with risk-reward ratio"""
    if action == "BUY":
        stop_loss = current_price - (atr * 1.5)
        take_profit = current_price + ((current_price - stop_loss) * rr_ratio)
    else:  # SELL
        stop_loss = current_price + (atr * 1.5)
        take_profit = current_price - ((stop_loss - current_price) * rr_ratio)
    
    return stop_loss, take_profit

def generate_combined_signal(tool_signals, current_price, symbol, interval, tool_weights, combined_settings):
    """Generate combined trading signal based on all tool signals with weighted scoring"""
    
    buy_signals = []
    sell_signals = []
    hold_signals = []
    
    confidence_threshold = combined_settings.get('confidence_threshold', 0.6)
    min_tool_agreement = combined_settings.get('min_tool_agreement', 2)
    rr_ratio_map = {"1:1": 1.0, "1:1.5": 1.5, "1:2": 2.0, "1:2.5": 2.5, "1:3": 3.0}
    selected_rr_ratio = rr_ratio_map.get(combined_settings.get('risk_reward_ratio', '1:2'), 2.0)
    
    # Collect signals from all tools
    for tool, signal_data in tool_signals.items():
        if tool == "gann":
            # Special handling for Gann's indirect information
            gann_analysis = analyze_gann_information(signal_data, current_price)
            action = 'BUY' if gann_analysis['bias'] == 'bullish' else 'SELL' if gann_analysis['bias'] == 'bearish' else 'HOLD'
            confidence_numeric = convert_confidence_to_numeric(gann_analysis['confidence'])
            reason = " | ".join(gann_analysis['reasons']) if gann_analysis['reasons'] else "No clear bias - Neutral position"
            signal_info = {
                'tool': tool,
                'action': action,
                'confidence': confidence_numeric,
                'reason': reason,
                'entry_price': current_price,
                'stop_loss': None,
                'take_profit': None,
                'weight': tool_weights.get(tool, 0.5),
                'is_direct': False,
                'confidence_original': gann_analysis['confidence'],  # This should be string
                'detailed_analysis': gann_analysis['detailed_analysis']
            }
            if action == 'BUY':
                buy_signals.append(signal_info)
            elif action == 'SELL':
                sell_signals.append(signal_info)
            else:
                hold_signals.append(signal_info)
            continue

        # Generalized handling for other tools (Fibonacci, Elliott, Ichimoku, Wyckoff)
        confidence_raw = signal_data.get('confidence', 'Low')
        if isinstance(confidence_raw, (int, float)):
            confidence_str = convert_numeric_to_confidence(confidence_raw)
        else:
            confidence_str = str(confidence_raw)
        confidence_numeric = convert_confidence_to_numeric(confidence_str)

        action = 'HOLD'
        reason = 'No clear signal from tool'
        entry_price = current_price
        stop_loss = None
        take_profit = None

        trade_action = signal_data.get('trade_action')
        signals = signal_data.get('signals', [])

        if tool == 'elliott':
            # Special aggregation for Elliott: collect all signals across degrees
            all_signals = []
            for degree_data in signal_data.get('wave_data_by_degree', {}).values():
                all_signals.extend(degree_data.get('signals', []))
            signals = all_signals  # Override with aggregated

        if trade_action:  # Fibonacci-style
            action = trade_action.upper()
            reason = ' | '.join(signal_data.get('signal_descriptions', ['No description provided']))
            entry_price = signal_data.get('entry_price', current_price)
            stop_loss = signal_data.get('stop_loss')
            take_profit = signal_data.get('take_profit')
        elif signals:  # Signal-based (Elliott, Ichimoku, Wyckoff)
            # Take the first signal (or aggregate if needed)
            signal = signals[0]
            if isinstance(signal, dict):
                # Dict format (assumed for Elliott-like)
                action_type = signal.get('type') or signal.get('action')
                if action_type:
                    action = action_type.upper()
                reason = signal.get('reason', 'No reason provided')
                entry_price = signal.get('entry_price', current_price)
                stop_loss = signal.get('sl') or signal.get('stop_loss')
                take_profit = signal.get('tp') or signal.get('take_profit')
            elif isinstance(signal, str):
                # String format (possible for Ichimoku/Wyckoff signals)
                reason = signal  # Use string as reason
                # Fallback to 'trend' or 'phase' for action
                trend = signal_data.get('trend', 'none').lower()
                phase = signal_data.get('phase', 'none').lower()
                if 'bullish' in trend or 'accumulation' in phase or 'markup' in phase:
                    action = 'BUY'
                elif 'bearish' in trend or 'distribution' in phase or 'markdown' in phase:
                    action = 'SELL'
        else:
            # No signals or trade_action: Fallback to trend/phase
            trend = signal_data.get('trend', 'none').lower()
            phase = signal_data.get('phase', 'none').lower()
            if 'bullish' in trend or 'accumulation' in phase or 'markup' in phase:
                action = 'BUY'
                reason = f"Bullish trend/phase: {trend or phase}"
            elif 'bearish' in trend or 'distribution' in phase or 'markdown' in phase:
                action = 'SELL'
                reason = f"Bearish trend/phase: {trend or phase}"

        signal_info = {
            'tool': tool,
            'action': action,
            'confidence': confidence_numeric,
            'reason': reason,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'weight': tool_weights.get(tool, 1.0),
            'is_direct': True,
            'confidence_original': confidence_str  # Always string for display
        }
        
        if action == "BUY":
            buy_signals.append(signal_info)
        elif action == "SELL":
            sell_signals.append(signal_info)
        else:
            hold_signals.append(signal_info)
    
    # Calculate weighted combined signal strength
    buy_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in buy_signals)
    sell_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in sell_signals)
    hold_strength = sum(float(sig['confidence']) * float(sig['weight']) for sig in hold_signals)
    
    # Only count direct tools for agreement (exclude Gann)
    direct_tools = [sig for sig in buy_signals + sell_signals + hold_signals if sig['is_direct']]
    total_direct_tools = len(set(sig['tool'] for sig in direct_tools))  # Number of tools that provided a signal
    buy_count = len([sig for sig in buy_signals if sig['is_direct']])
    sell_count = len([sig for sig in sell_signals if sig['is_direct']])
    hold_count = len([sig for sig in hold_signals if sig['is_direct']])
    
    # Calculate agreement percentage based on direct tools only
    total_signals = buy_count + sell_count + hold_count
    agreement_percentage = max(buy_count, sell_count, hold_count) / total_signals if total_signals > 0 else 0
    
    # Determine combined action with weighted scoring
    weighted_buy_score = buy_strength * agreement_percentage if buy_count > 0 else 0
    weighted_sell_score = sell_strength * agreement_percentage if sell_count > 0 else 0
    weighted_hold_score = hold_strength * agreement_percentage if hold_count > 0 else 0
    
    max_score = max(weighted_buy_score, weighted_sell_score, weighted_hold_score)
    
    reasons = []
    
    if max_score == weighted_buy_score and buy_count >= min_tool_agreement and weighted_buy_score >= confidence_threshold:
        combined_action = "BUY"
        combined_confidence_numeric = weighted_buy_score
        reasons = [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in buy_signals]
        agreement_level = 'high' if agreement_percentage >= 0.75 else 'medium' if agreement_percentage >= 0.5 else 'low'
        
    elif max_score == weighted_sell_score and sell_count >= min_tool_agreement and weighted_sell_score >= confidence_threshold:
        combined_action = "SELL"
        combined_confidence_numeric = weighted_sell_score
        reasons = [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in sell_signals]
        agreement_level = 'high' if agreement_percentage >= 0.75 else 'medium' if agreement_percentage >= 0.5 else 'low'
        
    else:
        # Default to HOLD if no clear buy/sell or if hold is strongest
        combined_action = "HOLD"
        combined_confidence_numeric = weighted_hold_score if weighted_hold_score > 0 else 0.5
        if total_signals == 0:
            reasons.append("No signals from any tools")
        if max(buy_count, sell_count) < min_tool_agreement:
            reasons.append(f"Insufficient tool agreement: {max(buy_count, sell_count)}/{min_tool_agreement} tools agree on BUY/SELL")
        if max(weighted_buy_score, weighted_sell_score) < confidence_threshold:
            reasons.append(f"Confidence score {max(weighted_buy_score, weighted_sell_score):.2f} below threshold {confidence_threshold}")
        if not reasons:
            reasons = ["Insufficient tool agreement or confidence"]
        reasons += [f"{sig['tool']} ({sig['weight']}x): {sig['reason']}" for sig in hold_signals]
        agreement_level = 'medium' if combined_confidence_numeric > 0.5 else 'low'
    
    # Convert numeric confidence back to string for display
    combined_confidence = convert_numeric_to_confidence(combined_confidence_numeric)
    
    # Calculate position sizing based on confidence and agreement
    if combined_action in ["BUY", "SELL"] and combined_confidence_numeric > 0:
        # Dynamic position sizing based on confidence and agreement
        base_size = min(combined_confidence_numeric * 100, 100)
        agreement_multiplier = 1.0 if agreement_level == 'high' else 0.7 if agreement_level == 'medium' else 0.5
        position_size = base_size * agreement_multiplier
        
        # Calculate stop loss and take profit with selected risk-reward ratio
        atr = tool_signals.get('fibonacci', {}).get('atr_value', current_price * 0.02)
        stop_loss, take_profit = calculate_tp_sl_rr(current_price, combined_action, atr, selected_rr_ratio)
        
    else:
        position_size = 0
        stop_loss = None
        take_profit = None
    
    return {
        'action': combined_action,
        'confidence': combined_confidence,  # This is now a string
        'confidence_numeric': combined_confidence_numeric,  # This is the numeric value
        'position_size': position_size,
        'entry_price': current_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'reasons': reasons,
        'agreement_level': agreement_level,
        'risk_reward_ratio': selected_rr_ratio,
        'tool_breakdown': {
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'hold_signals': hold_signals,
            'total_direct_tools': total_direct_tools,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'agreement_percentage': agreement_percentage
        },
        'current_price': current_price  # Add for manage_trades
    }

def run_scheduled_analysis():
    auto_state = load_auto_refresh_state()
    if not auto_state.get('enabled', False):
        return

    config = load_analysis_config()  # This loads the saved configuration
    combined_settings = load_combined_settings()
    symbols = config['symbols']  # Use the saved symbols
    interval = config['interval']
    candle_limit = config['candle_limit']
    selected_tools = config['selected_tools']
    enable_buy = config.get('enable_buy', True)
    enable_sell = config.get('enable_sell', True)

    # Rest of the function remains the same...
    # Load persistent trade data
    active_trades, trade_history = load_trade_data()
    session_data = {'active_trades': active_trades, 'trade_history': trade_history}

    tool_results = {}
    
    if 'fibonacci' in selected_tools:
        fib_analyses = run_analysis_for_tool('fibonacci', symbols, interval, candle_limit, config)
        manage_trades('fibonacci', fib_analyses, session_data, interval, enable_buy=enable_buy, enable_sell=enable_sell)
        tool_results['fibonacci'] = fib_analyses
    
    if 'elliott' in selected_tools:
        elliott_analyses = run_analysis_for_tool('elliott', symbols, interval, candle_limit, config)
        manage_trades('elliott', elliott_analyses, session_data, interval, enable_buy=enable_buy, enable_sell=enable_sell)
        tool_results['elliott'] = elliott_analyses
    
    if 'ichimoku' in selected_tools:
        ichimoku_analyses = run_analysis_for_tool('ichimoku', symbols, interval, candle_limit, config)
        manage_trades('ichimoku', ichimoku_analyses, session_data, interval, enable_buy=enable_buy, enable_sell=enable_sell)
        tool_results['ichimoku'] = ichimoku_analyses
    
    if 'wyckoff' in selected_tools:
        wyckoff_analyses = run_analysis_for_tool('wyckoff', symbols, interval, candle_limit, config)
        manage_trades('wyckoff', wyckoff_analyses, session_data, interval, enable_buy=enable_buy, enable_sell=enable_sell)
        tool_results['wyckoff'] = wyckoff_analyses
    
    if 'gann' in selected_tools:
        gann_analyses = run_analysis_for_tool('gann', symbols, interval, candle_limit, config)
        tool_results['gann'] = gann_analyses
    
    # Generate combined analysis (optional, if needed for trades)
    combined_analyses = {}
    for symbol in symbols:
        symbol_tool_signals = {tool: data[symbol] for tool, data in tool_results.items() if symbol in data}
        if symbol_tool_signals:
            current_price = next(iter(symbol_tool_signals.values()))['current_price']
            combined_signal = generate_combined_signal(
                symbol_tool_signals, current_price, symbol, interval,
                combined_settings['tool_weights'], combined_settings
            )
            combined_analyses[symbol] = combined_signal

    # Manage combined trades
    manage_trades('combined', combined_analyses, session_data, interval, enable_buy=enable_buy, enable_sell=enable_sell)

    # Save updated trade data
    save_trade_data(session_data['active_trades'], session_data['trade_history'])

    # Update last analysis time
    last_time = datetime.now(timezone.utc).isoformat()
    auto_state['last_analysis_time'] = last_time
    save_auto_refresh_state(auto_state)

    print(f"Scheduled analysis completed at {last_time}")

def get_running_days():
    running_data = load_running_time()
    start_time_str = running_data.get('start_time')
    if not start_time_str:
        return 0
    start_time = datetime.fromisoformat(start_time_str)
    elapsed = datetime.now(timezone.utc) - start_time
    return elapsed.days

@scheduler.task('interval', id='auto_analysis', minutes=1)  # Adjust interval as needed
def scheduled_task():
    run_scheduled_analysis()

@app.route('/', methods=['GET', 'POST'])
def index():
    init_session()
    
    popular_symbols = ["BTCUSDT","ZKCUSDT","DEGOUSDT","BELUSDT", "YBUSDT","ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT", "DOTUSDT", "DOGEUSDT", 
                       "LTCUSDT", "LINKUSDT", "AVAXUSDT", "UNIUSDT", "ATOMUSDT"]
    intervals = ["1m", "3m", "5m", "15m", "1h", "2h", "3h", "4h", "1d"]
    
    fibonacci_analyses = []
    elliott_analyses = {}
    ichimoku_analyses = {}
    wyckoff_analyses = {}
    gann_analyses = {}
    combined_analyses = {}
    
    # Handle auto-refresh state
    auto_refresh_enabled = session.get('auto_refresh_enabled', False)
    combined_settings = session.get('combined_settings', load_combined_settings())
    
    # Load the saved configuration to pre-populate form fields
    saved_config = load_analysis_config()
    
    # Initialize config with saved values for GET requests
    config = saved_config.copy()
    
    # Load budget data
    budget = load_budget()
    
    if request.method == 'POST':
        # Update auto-refresh state if checkbox was submitted
        auto_refresh_enabled = 'auto_refresh' in request.form
        session['auto_refresh_enabled'] = auto_refresh_enabled
        auto_state = {'enabled': auto_refresh_enabled, 'last_analysis_time': session.get('last_analysis_time')}
        save_auto_refresh_state(auto_state)
        
        # If enabling auto-refresh, set start time if not set
        running_data = load_running_time()
        if auto_refresh_enabled and not running_data.get('start_time'):
            running_data['start_time'] = datetime.now(timezone.utc).isoformat()
            save_running_time(running_data)
        elif not auto_refresh_enabled:
            # Optionally reset start_time when disabled
            running_data['start_time'] = None
            save_running_time(running_data)
        
        # Update combined settings
        if 'confidence_threshold' in request.form:
            combined_settings['confidence_threshold'] = float(request.form.get('confidence_threshold', 0.6))
            combined_settings['min_tool_agreement'] = int(request.form.get('min_tool_agreement', 2))
            combined_settings['risk_reward_ratio'] = request.form.get('risk_reward_ratio', '1:2')
            # Update tool weights
            for tool in ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann']:
                weight_key = f"{tool}_weight"
                if weight_key in request.form:
                    combined_settings['tool_weights'][tool] = float(request.form.get(weight_key, 1.0))
            
            session['combined_settings'] = combined_settings
            save_combined_settings(combined_settings)
        
        # Get selected tools and unified settings from the form
        selected_tools = request.form.getlist('tools') 
        if not selected_tools:
            selected_tools = ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann']
        
        symbols = request.form.getlist('symbols')
        interval = request.form.get('interval', '5m')
        
        # Add custom symbols
        custom_symbols_input = request.form.get('custom_symbols', '').upper()
        custom_symbols = [s.strip() for s in custom_symbols_input.split(',') if s.strip()]
        for custom_symbol in custom_symbols:
            if custom_symbol and custom_symbol not in symbols:
                symbols.append(custom_symbol)
        
        if not symbols:
            symbols = ['BTCUSDT']
        
        candle_limit = int(request.form.get('candle_limit', 1000))

        # Update config from form data
        config.update({
            'selected_tools': selected_tools,
            'symbols': symbols,
            'interval': interval,
            'candle_limit': candle_limit,
            'fib_window': int(request.form.get('window', 60)),
            'fib_threshold': float(request.form.get('fib_threshold', 0.3)) / 100,
            'elliott_thresholds': {
                'Minor': float(request.form.get('minor_threshold', 0.5)) / 100,
                'Intermediate': float(request.form.get('intermediate_threshold', 1.5)) / 100,
                'Major': float(request.form.get('major_threshold', 3.0)) / 100
            },
            'elliott_degrees': request.form.getlist('elliott_degrees') or ['Minor', 'Intermediate', 'Major'],
            'use_smoothing': 'use_smoothing' in request.form,
            'smooth_period': int(request.form.get('smooth_period', 3)),
            'show_ema': 'show_ema' in request.form,
            'show_bb': 'show_bb' in request.form,
            'show_volume': 'show_volume' in request.form,
            'show_rsi': 'show_rsi' in request.form,
            'show_macd': 'show_macd' in request.form,
            'show_ema_ichimoku': 'show_ema_ichimoku' in request.form,
            'show_bb_ichimoku': 'show_bb_ichimoku' in request.form,
            'show_volume_ichimoku': 'show_volume_ichimoku' in request.form,
            'show_rsi_ichimoku': 'show_rsi_ichimoku' in request.form,
            'show_macd_ichimoku': 'show_macd_ichimoku' in request.form,
            'show_ema_wyckoff': 'show_ema_wyckoff' in request.form,
            'show_bb_wyckoff': 'show_bb_wyckoff' in request.form,
            'show_volume_wyckoff': 'show_volume_wyckoff' in request.form,
            'show_rsi_wyckoff': 'show_rsi_wyckoff' in request.form,
            'show_macd_wyckoff': 'show_macd_wyckoff' in request.form,
            'gann_subtools': request.form.getlist('gann_subtools') or ['Gann Fan', 'Gann Square', 'Gann Box', 'Gann Square Fixed'],
            'pivot_choice': request.form.get('pivot_choice', 'Auto (based on trend)'),
            'show_ema_gann': 'show_ema_gann' in request.form,
            'show_bb_gann': 'show_bb_gann' in request.form,
            'show_volume_gann': 'show_volume_gann' in request.form,
            'show_rsi_gann': 'show_rsi_gann' in request.form,
            'show_macd_gann': 'show_macd_gann' in request.form,
            'enable_buy': 'enable_buy' in request.form,  # New
            'enable_sell': 'enable_sell' in request.form  # New
        })
        save_analysis_config(config)

        # Run analyses for all selected tools
        tool_results = {}
        
        if 'fibonacci' in selected_tools:
            fibonacci_analyses_dict = run_analysis_for_tool('fibonacci', symbols, interval, candle_limit, config)
            fibonacci_analyses = list(fibonacci_analyses_dict.values())
            manage_trades('fibonacci', fibonacci_analyses_dict, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
            tool_results['fibonacci'] = fibonacci_analyses_dict
        
        if 'elliott' in selected_tools:
            elliott_analyses = run_analysis_for_tool('elliott', symbols, interval, candle_limit, config)
            manage_trades('elliott', elliott_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
            tool_results['elliott'] = elliott_analyses
        
        if 'ichimoku' in selected_tools:
            ichimoku_analyses = run_analysis_for_tool('ichimoku', symbols, interval, candle_limit, config)
            manage_trades('ichimoku', ichimoku_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
            tool_results['ichimoku'] = ichimoku_analyses
        
        if 'wyckoff' in selected_tools:
            wyckoff_analyses = run_analysis_for_tool('wyckoff', symbols, interval, candle_limit, config)
            manage_trades('wyckoff', wyckoff_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
            tool_results['wyckoff'] = wyckoff_analyses
        
        if 'gann' in selected_tools:
            gann_analyses = run_analysis_for_tool('gann', symbols, interval, candle_limit, config)
            tool_results['gann'] = gann_analyses
        
        # Generate combined analysis for each symbol
        for symbol in symbols:
            symbol_tool_signals = {}
            for tool_name, tool_data in tool_results.items():
                if symbol in tool_data:
                    symbol_tool_signals[tool_name] = tool_data[symbol]
            
            if symbol_tool_signals:
                current_price = next(iter(symbol_tool_signals.values()))['current_price']
                combined_signal = generate_combined_signal(
                    symbol_tool_signals, 
                    current_price, 
                    symbol, 
                    interval, 
                    combined_settings['tool_weights'],
                    combined_settings
                )
                combined_analyses[symbol] = combined_signal
        
        # Manage combined trades
        manage_trades('combined', combined_analyses, session, interval, enable_buy=config['enable_buy'], enable_sell=config['enable_sell'])
        
        # Save trade data to JSON files
        save_trade_data(session['active_trades'], session['trade_history'])
        
        # Update last analysis time for auto-refresh
        session['last_analysis_time'] = datetime.now(timezone.utc).isoformat()
        save_auto_refresh_state({
            'enabled': auto_refresh_enabled,
            'last_analysis_time': session['last_analysis_time']
        })
        
        # Reload budget after trades
        budget = load_budget()
        session.modified = True
    
    # For GET requests or auto-refresh, use saved config to pre-populate form
    else:
        # Use the saved configuration for form pre-population
        config = saved_config
        symbols = config.get('symbols', ['BTCUSDT'])
        selected_tools = config.get('selected_tools', ['fibonacci', 'elliott', 'ichimoku', 'wyckoff', 'gann'])
    
    # Load trade data from JSON files (ensures consistency)
    active_trades, trade_history = load_trade_data()
    session['active_trades'] = active_trades
    session['trade_history'] = trade_history
    
    # Calculate wins/losses for each analysis type
    fibonacci_wins = sum(1 for t in trade_history.get('fibonacci', []) if t.get('outcome') == 'win')
    fibonacci_losses = len(trade_history.get('fibonacci', [])) - fibonacci_wins
    
    elliott_wins = sum(1 for t in trade_history.get('elliott', []) if t.get('outcome') == 'win')
    elliott_losses = len(trade_history.get('elliott', [])) - elliott_wins
    
    ichimoku_wins = sum(1 for t in trade_history.get('ichimoku', []) if t.get('outcome') == 'win')
    ichimoku_losses = len(trade_history.get('ichimoku', [])) - ichimoku_wins
    
    wyckoff_wins = sum(1 for t in trade_history.get('wyckoff', []) if t.get('outcome') == 'win')
    wyckoff_losses = len(trade_history.get('wyckoff', [])) - wyckoff_wins
    
    combined_wins = sum(1 for t in trade_history.get('combined', []) if t.get('outcome') == 'win')
    combined_losses = len(trade_history.get('combined', [])) - combined_wins

    # Get running days
    running_days = get_running_days()

    return render_template('index.html', 
                         fibonacci_analyses=fibonacci_analyses,
                         elliott_analyses=elliott_analyses,
                         ichimoku_analyses=ichimoku_analyses,
                         wyckoff_analyses=wyckoff_analyses,
                         gann_analyses=gann_analyses,
                         combined_analyses=combined_analyses,
                         active_trades=active_trades,
                         trade_history=trade_history,
                         fibonacci_wins=fibonacci_wins,
                         fibonacci_losses=fibonacci_losses,
                         elliott_wins=elliott_wins,
                         elliott_losses=elliott_losses,
                         ichimoku_wins=ichimoku_wins,
                         ichimoku_losses=ichimoku_losses,
                         wyckoff_wins=wyckoff_wins,
                         wyckoff_losses=wyckoff_losses,
                         combined_wins=combined_wins,
                         combined_losses=combined_losses,
                         popular_symbols=popular_symbols,
                         intervals=intervals,
                         auto_refresh_enabled=auto_refresh_enabled,
                         combined_settings=combined_settings,
                         running_days=running_days,
                         budget=budget,  # Make sure budget is passed
                         config=config,
                         zip=zip)

@app.route('/refresh_price/<symbol>')
def refresh_price(symbol):
    from utils import fetch_current_price
    try:
        price = fetch_current_price(symbol)
        return jsonify({'price': price})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/close_trade', methods=['POST'])
def close_trade():
    try:
        data = request.get_json()
        tool = data['tool']
        symbol = data['symbol']
        degree = data.get('degree')
        trade_key = f"{symbol}_{degree}" if tool == 'elliott' else symbol
        
        if trade_key in session['active_trades'][tool]:
            trade = session['active_trades'][tool][trade_key]
            close_price = float(data['close_price'])
            outcome = data['outcome']
            
            # Calculate fees and profit/loss
            closing_fee, net_proceeds = calculate_trade_costs(trade['invested_amount'], is_opening=False)
            
            if trade['action'] == "BUY":
                gross_profit = (close_price - trade['entry_price']) / trade['entry_price'] * 100
                gross_profit_usd = (close_price - trade['entry_price']) * (trade['net_investment'] / trade['entry_price'])
                net_profit_usd = gross_profit_usd - trade['entry_fee'] - closing_fee
            else:  # SELL
                gross_profit = (trade['entry_price'] - close_price) / trade['entry_price'] * 100
                gross_profit_usd = (trade['entry_price'] - close_price) * (trade['net_investment'] / trade['entry_price'])
                net_profit_usd = gross_profit_usd - trade['entry_fee'] - closing_fee
            
            net_profit_percent = (net_profit_usd / trade['invested_amount']) * 100
            
            closed_trade = trade.copy()
            closed_trade.update({
                'outcome': outcome,
                'close_price': close_price,
                'profit_pct': gross_profit,
                'net_profit_pct': net_profit_percent,
                'net_profit_usd': net_profit_usd,
                'gross_profit_usd': gross_profit_usd,
                'closing_fee': closing_fee,
                'total_fees': trade['entry_fee'] + closing_fee,
                'close_time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'interval': trade['interval']
            })
            session['trade_history'][tool].append(closed_trade)
            
            # Return budget when trade is closed manually
            update_budget(trade['invested_amount'], closing_fee, "return", gross_profit_usd=gross_profit_usd)
            
            del session['active_trades'][tool][trade_key]
            
            # Save to JSON files
            save_trade_data(session['active_trades'], session['trade_history'])
            session.modified = True
            
            print(f"{tool.capitalize()} trade closed for {symbol}: {outcome}, Net Profit: ${net_profit_usd:.2f}")
            return jsonify({'status': 'success'})
        else:
            return jsonify({'status': 'error', 'message': 'Trade not found'}), 404
    except Exception as e:
        print(f"Error closing trade: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500    

@app.route('/update_budget', methods=['POST'])
def update_budget_route():
    try:
        data = request.get_json()
        new_budget = float(data['total_budget'])
        
        budget = load_budget()
        budget['total_budget'] = new_budget
        budget['remaining_budget'] = new_budget - budget['used_budget']
        budget['initial_budget'] = new_budget
        
        save_budget(budget)
        session['budget'] = budget
        
        return jsonify({'status': 'success', 'budget': budget})
    except Exception as e:
        print(f"Error updating budget: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500
    

@app.route('/reset_budget', methods=['POST'])
def reset_budget():
    """Reset budget to initial state"""
    try:
        default_budget = {
            'total_budget': 5000.0,
            'used_budget': 0.0,
            'remaining_budget': 5000.0,
            'initial_budget': 5000.0,
            'total_fees': 0.0,
            'total_invested': 0.0
        }
        save_budget(default_budget)
        session['budget'] = default_budget
        return jsonify({'status': 'success', 'budget': default_budget})
    except Exception as e:
        print(f"Error resetting budget: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500 

@app.route('/auto_refresh_status')
def auto_refresh_status():
    """Check if auto-refresh should trigger new analysis"""
    auto_refresh_state = load_auto_refresh_state()
    should_refresh = auto_refresh_state.get('enabled', False)
    budget = load_budget()
    
    return jsonify({
        'enabled': should_refresh,
        'should_refresh': should_refresh,
        'last_analysis_time': auto_refresh_state.get('last_analysis_time'),
        'budget': budget  # Add budget to response
    })

@app.route('/running_days')
def running_days():
    return jsonify({'running_days': get_running_days()})
    
    
    
if not scheduler.running:
    scheduler.start()

















