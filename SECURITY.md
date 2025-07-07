# Security Policy

## Supported Versions

We actively maintain and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.2.x   | :white_check_mark: |
| 0.1.x   | :x:                |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue, please follow these steps:

### 1. **DO NOT** create a public GitHub issue
Security vulnerabilities should be reported privately to prevent exploitation.

### 2. **Email us directly**
Send a detailed report to: `security@pepino.dev` (or your preferred security contact)

### 3. **Include in your report**
- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** assessment
- **Suggested fix** (if you have one)
- **Your contact information** for follow-up

### 4. **What happens next**
- We'll acknowledge receipt within 48 hours
- We'll investigate and provide updates
- We'll work on a fix and coordinate disclosure
- We'll credit you in the security advisory (if desired)

## Security Considerations

### Discord Bot Security
- **Token Protection**: Never commit Discord tokens to version control
- **Permission Scoping**: Use minimum required permissions
- **Rate Limiting**: Respect Discord API rate limits
- **Data Privacy**: Only collect necessary data

### Data Security
- **Local Storage**: Data stored locally in SQLite database
- **No Cloud Storage**: No data sent to external services
- **User Control**: Users control their own data
- **Encryption**: Consider encrypting sensitive databases

### Code Security
- **Dependency Updates**: Regular security updates
- **Input Validation**: Comprehensive input sanitization
- **SQL Injection Prevention**: Parameterized queries only
- **Error Handling**: No sensitive data in error messages

## Best Practices for Users

### Environment Security
```bash
# Use environment variables for sensitive data
export DISCORD_TOKEN="your_token_here"
export DATABASE_URL="sqlite:///secure_path/database.db"

# Set proper file permissions
chmod 600 .env
chmod 600 discord_messages.db
```

### Network Security
- **HTTPS Only**: Use HTTPS for any external communications
- **Firewall Rules**: Restrict access to Discord API only
- **VPN Usage**: Consider VPN for additional privacy

### Data Protection
- **Regular Backups**: Backup your database regularly
- **Access Control**: Limit who can access your analysis data
- **Data Retention**: Consider data retention policies

## Security Features

### Built-in Protections
- **Base Filtering**: Automatic exclusion of bot messages and test channels
- **Input Validation**: Pydantic validation for all configuration
- **Error Sanitization**: No sensitive data in error messages
- **Logging Security**: No sensitive data in logs

### Configuration Security
```python
# Secure configuration example
class Settings(BaseSettings):
    discord_token: str = Field(..., description="Discord bot token")
    database_url: str = Field(default="sqlite:///data/discord_messages.db")
    
    # Security validators
    @field_validator("discord_token")
    @classmethod
    def validate_token(cls, v):
        if not v or len(v) < 10:
            raise ValueError("Invalid Discord token")
        return v
```

## Vulnerability Disclosure

When we discover or receive reports of security vulnerabilities:

1. **Immediate Assessment**: Evaluate severity and impact
2. **Fix Development**: Create and test security patches
3. **Coordinated Disclosure**: Release fix with security advisory
4. **User Notification**: Notify users through appropriate channels
5. **Post-Mortem**: Document lessons learned

## Security Updates

### Automatic Updates
- **Dependency Scanning**: Regular security scans of dependencies
- **Vulnerability Monitoring**: Monitor for known vulnerabilities
- **Update Notifications**: Notify users of security updates

### Manual Updates
```bash
# Update dependencies
poetry update

# Check for security issues
poetry audit

# Update to latest version
git pull origin main
poetry install
```

## Responsible Disclosure

We follow responsible disclosure practices:

- **Private Reporting**: Security issues reported privately
- **Timely Response**: Acknowledge and investigate promptly
- **Coordinated Release**: Release fixes with proper disclosure
- **Credit Attribution**: Credit security researchers appropriately
- **No Retaliation**: Welcome security research and feedback

## Security Contacts

- **Primary**: `security@pepino.dev`
- **Backup**: GitHub Security Advisories
- **Response Time**: 48 hours for acknowledgment

## Security Resources

- [Discord Security Best Practices](https://discord.com/developers/docs/topics/security)
- [Python Security](https://python-security.readthedocs.io/)
- [OWASP Security Guidelines](https://owasp.org/www-project-top-ten/)

Thank you for helping keep Pepino secure! 🔒 