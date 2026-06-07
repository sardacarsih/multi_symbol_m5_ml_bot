import argparse

from config import SYMBOLS


def configured_symbols() -> list[str]:
    return list(SYMBOLS.keys())


def validate_symbol(symbol: str) -> str:
    normalized = symbol.upper()
    if normalized not in SYMBOLS:
        raise ValueError(f"Unknown symbol '{symbol}'. Available: {', '.join(configured_symbols())}")
    return normalized


def parse_symbol_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--symbol", help="Logical symbol, for example XAUUSD")
    group.add_argument("--symbols", nargs="+", help="Logical symbols, for example XAUUSD USTEC")
    group.add_argument("--all", action="store_true", help="Run all configured symbols")
    return parser


def resolve_symbols(args) -> list[str]:
    if getattr(args, "all", False) or (not getattr(args, "symbol", None) and not getattr(args, "symbols", None)):
        return configured_symbols()
    if getattr(args, "symbol", None):
        return [validate_symbol(args.symbol)]
    return [validate_symbol(symbol) for symbol in args.symbols]
