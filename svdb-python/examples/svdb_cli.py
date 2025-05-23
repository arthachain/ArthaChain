#!/usr/bin/env python3
"""
SVDB Command Line Interface (CLI)

A developer tool for interacting with the State Variable Database
"""

import os
import sys
import asyncio
import argparse
import json
from typing import Optional, List, Dict, Any

try:
    from svdb import SvdbClient
except ImportError:
    print("Error: svdb package not found. Please install it with:")
    print("  pip install svdb")
    sys.exit(1)


class SvdbCli:
    def __init__(self):
        self.client = None
        self.parser = self._build_parser()

    def _build_parser(self) -> argparse.ArgumentParser:
        """Build the command line argument parser"""
        parser = argparse.ArgumentParser(
            description="SVDB Command Line Interface - Interact with State Variable Database",
            formatter_class=argparse.RawTextHelpFormatter
        )

        # Global options
        parser.add_argument('--db-path', '-d', 
                           default=os.environ.get('SVDB_PATH', './data/svdb'),
                           help='Path to SVDB database (default: ./data/svdb)')
        parser.add_argument('--encrypt-key', '-e',
                           help='Path to file containing 32-byte encryption key')
        parser.add_argument('--blockchain', '-b',
                           help='Blockchain endpoint URL (default: none)')
        parser.add_argument('--format', '-f',
                           choices=['text', 'json'],
                           default='text',
                           help='Output format (default: text)')

        # Create subparsers for commands
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')

        # Store command
        store_parser = subparsers.add_parser('store', help='Store data')
        store_parser.add_argument('key', help='Key to store data under')
        store_parser.add_argument('value', help='Data to store (string or @file to read from file)')
        store_parser.add_argument('--encrypt', '-e', action='store_true', 
                               help='Encrypt the data')

        # Retrieve command
        retrieve_parser = subparsers.add_parser('retrieve', help='Retrieve data')
        retrieve_parser.add_argument('key', help='Key to retrieve')
        retrieve_parser.add_argument('--output', '-o', 
                                 help='Output file (default: stdout)')

        # Delete command
        delete_parser = subparsers.add_parser('delete', help='Delete data')
        delete_parser.add_argument('key', help='Key to delete')

        # Exists command
        exists_parser = subparsers.add_parser('exists', help='Check if key exists')
        exists_parser.add_argument('key', help='Key to check')

        # Verify command
        verify_parser = subparsers.add_parser('verify', help='Verify data integrity')
        verify_parser.add_argument('key', help='Key to verify')
        verify_parser.add_argument('hash', help='Expected hash value')

        # List command
        list_parser = subparsers.add_parser('list', help='List keys with prefix')
        list_parser.add_argument('prefix', nargs='?', default='', 
                              help='Prefix to search for (default: list all)')

        # Batch commands
        batch_store_parser = subparsers.add_parser('batch-store', 
                                               help='Store multiple items')
        batch_store_parser.add_argument('items_file', 
                                     help='JSON file with items to store (format: [{"key": "k", "value": "v", "encrypt": false}, ...']')

        batch_retrieve_parser = subparsers.add_parser('batch-retrieve', 
                                                  help='Retrieve multiple items')
        batch_retrieve_parser.add_argument('keys', 
                                       help='Comma-separated keys or @file with one key per line')
        batch_retrieve_parser.add_argument('--output', '-o', 
                                       help='Output file for results (default: stdout)')

        batch_delete_parser = subparsers.add_parser('batch-delete', 
                                                help='Delete multiple items')
        batch_delete_parser.add_argument('keys', 
                                     help='Comma-separated keys or @file with one key per line')

        return parser

    async def _init_client(self, args: argparse.Namespace) -> None:
        """Initialize the SVDB client based on command line arguments"""
        encryption_key = None
        if args.encrypt_key:
            try:
                with open(args.encrypt_key, 'rb') as f:
                    encryption_key = f.read()
                    if len(encryption_key) != 32:
                        print(f"Error: Encryption key must be 32 bytes (got {len(encryption_key)})")
                        sys.exit(1)
            except Exception as e:
                print(f"Error reading encryption key: {e}")
                sys.exit(1)

        self.client = SvdbClient(args.db_path, encryption_key, args.blockchain)

    def _parse_value(self, value: str) -> bytes:
        """Parse a value from command line or file"""
        if value.startswith('@'):
            filename = value[1:]
            try:
                with open(filename, 'rb') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading from file {filename}: {e}")
                sys.exit(1)
        return value.encode('utf-8')

    def _parse_keys(self, keys_arg: str) -> List[str]:
        """Parse keys from comma-separated list or file"""
        if keys_arg.startswith('@'):
            filename = keys_arg[1:]
            try:
                with open(filename, 'r') as f:
                    return [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading keys from file {filename}: {e}")
                sys.exit(1)
        return [k.strip() for k in keys_arg.split(',') if k.strip()]

    def _output_result(self, result: Any, args: argparse.Namespace, output_file: Optional[str] = None) -> None:
        """Output result in the specified format"""
        if args.format == 'json':
            # Convert bytes to base64 for JSON output
            if isinstance(result, bytes):
                import base64
                result = base64.b64encode(result).decode('ascii')
            elif isinstance(result, dict):
                for k, v in result.items():
                    if isinstance(v, bytes):
                        result[k] = base64.b64encode(v).decode('ascii')
            
            json_output = json.dumps(result, indent=2)
            if output_file:
                with open(output_file, 'w') as f:
                    f.write(json_output)
            else:
                print(json_output)
        else:
            # Text format
            if output_file and isinstance(result, bytes):
                with open(output_file, 'wb') as f:
                    f.write(result)
            elif output_file:
                with open(output_file, 'w') as f:
                    f.write(str(result))
            else:
                if isinstance(result, bytes):
                    try:
                        # Try to decode as utf-8
                        print(result.decode('utf-8'))
                    except UnicodeDecodeError:
                        # Fall back to hex representation
                        print(f"Binary data: {result.hex()}")
                elif isinstance(result, dict):
                    for k, v in result.items():
                        if isinstance(v, bytes):
                            try:
                                v = v.decode('utf-8')
                            except UnicodeDecodeError:
                                v = f"Binary data: {v.hex()}"
                        print(f"{k}: {v}")
                elif isinstance(result, list):
                    for item in result:
                        print(item)
                else:
                    print(result)

    async def _handle_store(self, args: argparse.Namespace) -> None:
        """Handle store command"""
        data = self._parse_value(args.value)
        hash_value = await self.client.store(args.key, data, args.encrypt)
        self._output_result({"key": args.key, "hash": hash_value, "size": len(data)}, args)

    async def _handle_retrieve(self, args: argparse.Namespace) -> None:
        """Handle retrieve command"""
        try:
            data = await self.client.retrieve(args.key)
            self._output_result(data, args, args.output)
        except KeyError:
            print(f"Error: Key '{args.key}' not found")
            sys.exit(1)

    async def _handle_delete(self, args: argparse.Namespace) -> None:
        """Handle delete command"""
        await self.client.delete(args.key)
        self._output_result({"result": "success", "message": f"Key '{args.key}' deleted"}, args)

    async def _handle_exists(self, args: argparse.Namespace) -> None:
        """Handle exists command"""
        exists = await self.client.exists(args.key)
        self._output_result(exists, args)
        # Set exit code based on existence (useful for scripts)
        if not exists:
            sys.exit(1)

    async def _handle_verify(self, args: argparse.Namespace) -> None:
        """Handle verify command"""
        try:
            valid = await self.client.verify(args.key, args.hash)
            self._output_result(valid, args)
            if not valid:
                sys.exit(1)
        except Exception as e:
            print(f"Error verifying data: {e}")
            sys.exit(1)
    
    async def _handle_list(self, args: argparse.Namespace) -> None:
        """Handle list command - this requires extending the Python bindings, so stub for now"""
        print("Error: The 'list' command requires extending the Python bindings")
        print("Consider using the Rust SDK directly for this functionality")
        sys.exit(1)

    async def _handle_batch_store(self, args: argparse.Namespace) -> None:
        """Handle batch store command"""
        try:
            with open(args.items_file, 'r') as f:
                items_data = json.load(f)
        except Exception as e:
            print(f"Error reading items file: {e}")
            sys.exit(1)

        # Convert items to the format expected by the SDK
        batch_items = []
        for item in items_data:
            if 'key' not in item or 'value' not in item:
                print(f"Error: Each item must have 'key' and 'value' fields")
                sys.exit(1)
            
            if isinstance(item['value'], str) and item['value'].startswith('@'):
                value = self._parse_value(item['value'])
            else:
                value = item['value'].encode('utf-8') if isinstance(item['value'], str) else item['value']
            
            batch_items.append((
                item['key'],
                value,
                item.get('encrypt', False)
            ))

        results = await self.client.batch_store(batch_items)
        self._output_result(dict(results), args)

    async def _handle_batch_retrieve(self, args: argparse.Namespace) -> None:
        """Handle batch retrieve command"""
        keys = self._parse_keys(args.keys)
        try:
            results = await self.client.batch_retrieve(keys)
            self._output_result(dict(results), args, args.output)
        except Exception as e:
            print(f"Error retrieving batch: {e}")
            sys.exit(1)

    async def _handle_batch_delete(self, args: argparse.Namespace) -> None:
        """Handle batch delete command"""
        keys = self._parse_keys(args.keys)
        try:
            await self.client.batch_delete(keys)
            self._output_result({"result": "success", "message": f"Deleted {len(keys)} keys"}, args)
        except Exception as e:
            print(f"Error deleting batch: {e}")
            sys.exit(1)

    async def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with the given arguments"""
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return 1

        await self._init_client(parsed_args)

        # Dispatch to the appropriate handler
        command_handlers = {
            'store': self._handle_store,
            'retrieve': self._handle_retrieve,
            'delete': self._handle_delete,
            'exists': self._handle_exists,
            'verify': self._handle_verify,
            'list': self._handle_list,
            'batch-store': self._handle_batch_store,
            'batch-retrieve': self._handle_batch_retrieve,
            'batch-delete': self._handle_batch_delete,
        }

        handler = command_handlers.get(parsed_args.command)
        if handler:
            await handler(parsed_args)
            return 0
        else:
            self.parser.print_help()
            return 1


def main():
    """Main entry point"""
    cli = SvdbCli()
    try:
        sys.exit(asyncio.run(cli.run()))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 