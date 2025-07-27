# {{plugin_name_pascal}} Provider for PyneCore

A PyneCore data provider plugin for accessing {{plugin_name}} market data.

## Installation

```bash
pip install pynecore-{{plugin_name_kebab}}-provider
```

## Configuration

Add the following configuration to your `providers.toml` file:

```toml
[{{plugin_name_snake}}]
# Add your provider-specific configuration here
# api_key = "your_api_key_here"
# api_secret = "your_api_secret_here"
# base_url = "https://api.example.com"
```

## Usage

Once installed, the provider will be automatically discovered by PyneCore. You can use it with the CLI:

```bash
# List available symbols
pyne data download --provider {{plugin_name_snake}} --list-symbols

# Download OHLCV data
pyne data download --provider {{plugin_name_snake}} --symbol EURUSD --timeframe 1h --time-from 2024-01-01 --time-to 2024-01-31
```

Or programmatically:

```python
from pathlib import Path
from {{plugin_name_snake}}_provider import {{plugin_name_pascal}}Provider
from datetime import datetime, timezone

# Initialize provider
provider = {{plugin_name_pascal}}Provider(
    symbol="EURUSD",
    timeframe="1h",
    workdir=Path("./data")
)

# Download data
data = provider.download_ohlcv(
    symbol="EURUSD",
    timeframe="1h",
    time_from=datetime(2024, 1, 1, tzinfo=timezone.utc),
    time_to=datetime(2024, 1, 31, tzinfo=timezone.utc)
)

print(f"Downloaded {len(data)} candles")
if data:
    print(f"First candle: {data[0]}")
    print(f"Last candle: {data[-1]}")
```

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/pynecore-{{plugin_name_kebab}}-provider.git
cd pynecore-{{plugin_name_kebab}}-provider

# Install in development mode
pip install -e ".[dev]"
```

### Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov={{plugin_name_snake}}_provider
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
mypy src/
```

## Implementation Notes

This provider template includes:

- ✅ Complete PyneCore Provider interface implementation
- ✅ Proper entry point configuration for auto-discovery
- ✅ Configuration management through `providers.toml`
- ✅ Timeframe conversion utilities
- ✅ Symbol information and trading hours support
- ✅ Progress callback support for data downloads
- ✅ Comprehensive error handling
- ✅ Type hints and documentation
- ✅ Development tools configuration

### TODO: Customize for Your Provider

1. **Update API Integration**: Replace placeholder implementations in `provider.py` with actual API calls
2. **Configure Authentication**: Add required API keys/secrets to configuration
3. **Implement Timeframe Mapping**: Update timeframe conversion methods for your provider's format
4. **Add Symbol Mapping**: Implement symbol listing and information retrieval
5. **Handle Rate Limiting**: Add appropriate rate limiting and retry logic
6. **Add Tests**: Create comprehensive test suite for your provider
7. **Update Documentation**: Customize this README with provider-specific information

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.