#!/usr/bin/env python3

import os
import argparse
import sys
from web.app import app

def run_web_app(host='127.0.0.1', port=5000, debug=False):
    """Run the web application."""
    print(f"Starting Blackjack Strategy Generator web app on http://{host}:{port}/")
    app.run(host=host, port=port, debug=debug)

def run_tests():
    """Run all unit tests."""
    import unittest
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    return result.wasSuccessful()

def main():
    parser = argparse.ArgumentParser(description='Blackjack Strategy Generator')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Web app command
    web_parser = subparsers.add_parser('web', help='Run the web application')
    web_parser.add_argument('--host', type=str, default='127.0.0.1', help='Host to run the web app on')
    web_parser.add_argument('--port', type=int, default=5000, help='Port to run the web app on')
    web_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run unit tests')
    
    args = parser.parse_args()
    
    if args.command == 'web':
        run_web_app(host=args.host, port=args.port, debug=args.debug)
    elif args.command == 'test':
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()