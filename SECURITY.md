# Security

## Security Considerations

MemoryPipe stores data locally in SQLite. Here are security considerations:

### Data Storage
- All data is stored locally in SQLite — no data leaves your machine
- Database file should be treated as sensitive (contains user facts, preferences, conversations)
- Use file system permissions to restrict access to the database file

### Input Validation
- All CLI inputs are validated through Click's type system
- Text content is stored as-is — no code execution on stored content
- File paths in config are validated for existence before use

### No Network Dependencies
- Core functionality has zero network dependencies
- Vector search uses local TF-IDF (no external API calls)
- Optional embedding integrations should be added with proper credential management

### Best Practices
1. Set restrictive file permissions on your database file: `chmod 600 memory_pipe.db`
2. Use environment variables for sensitive data in config files
3. Don't store credentials or tokens in memory items
4. Use `.env` files (never commit them) for any optional API keys
5. Regularly run `memory-pipe cleanup` to remove expired data
