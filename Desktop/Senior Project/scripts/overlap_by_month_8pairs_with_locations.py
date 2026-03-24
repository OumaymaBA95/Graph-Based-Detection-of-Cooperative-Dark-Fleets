#!/usr/bin/env python3
"""
Variant of overlap_by_month_8pairs.py that also records approximate locations:
- mean_lat/lon for each vessel per month
- meet_lat/meet_lon (midpoint) for the pair per month
"""
from overlap_by_month_8pairs import main as _main


if __name__ == '__main__':
    _main()

