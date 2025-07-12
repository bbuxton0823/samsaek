"""
A2A Protocol Security Implementation
Handles encryption, signatures, and authentication for A2A messages
"""

import base64
import hashlib
import hmac
import json
import os
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import jwt

from ...core.config import config
from ...core.logging import get_logger, SamsaekLoggerMixin
from ...core.exceptions import A2AEncryptionError, A2ASignatureError, SecurityError
from .models import A2AMessage, A2ASecurityLevel


class A2ACryptoManager(SamsaekLoggerMixin):
    """
    Cryptographic manager for A2A protocol security
    Handles encryption, decryption, signing, and verification
    """
    
    def __init__(self, agent_id: str, private_key: Optional[str] = None, public_key: Optional[str] = None):
        super().__init__()
        self.agent_id = agent_id
        self.backend = default_backend()
        
        # Initialize keys
        self.private_key = None
        self.public_key = None
        self.symmetric_key = None
        
        # Key storage
        self.agent_keys: Dict[str, Any] = {}
        self.trusted_agents: Dict[str, str] = {}
        
        # Initialize cryptographic keys
        self._initialize_keys(private_key, public_key)
        
        self.logger.info("A2A crypto manager initialized", agent_id=agent_id)
    
    def _initialize_keys(self, private_key_pem: Optional[str] = None, public_key_pem: Optional[str] = None):
        """Initialize cryptographic keys"""
        try:
            if private_key_pem and public_key_pem:
                # Load provided keys
                self.private_key = serialization.load_pem_private_key(
                    private_key_pem.encode(),
                    password=None,
                    backend=self.backend
                )
                self.public_key = serialization.load_pem_public_key(
                    public_key_pem.encode(),
                    backend=self.backend
                )
            else:
                # Generate new key pair
                self.private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                    backend=self.backend
                )
                self.public_key = self.private_key.public_key()
            
            # Generate symmetric key for faster encryption
            self.symmetric_key = os.urandom(32)  # 256-bit key
            
            self.logger.info("Cryptographic keys initialized successfully")
            
        except Exception as e:
            raise A2AEncryptionError(
                "Failed to initialize cryptographic keys",
                context={"agent_id": self.agent_id, "error": str(e)}
            )
    
    def get_public_key_pem(self) -> str:
        """Get public key in PEM format"""
        try:
            pem = self.public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            return pem.decode()
        except Exception as e:
            raise A2AEncryptionError(
                "Failed to export public key",
                context={"agent_id": self.agent_id, "error": str(e)}
            )
    
    def get_private_key_pem(self) -> str:
        """Get private key in PEM format"""
        try:
            pem = self.private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            )
            return pem.decode()
        except Exception as e:
            raise A2AEncryptionError(
                "Failed to export private key",
                context={"agent_id": self.agent_id, "error": str(e)}
            )
    
    def add_trusted_agent(self, agent_id: str, public_key_pem: str):
        """Add a trusted agent's public key"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem.encode(),
                backend=self.backend
            )
            self.agent_keys[agent_id] = public_key
            self.trusted_agents[agent_id] = public_key_pem
            
            self.logger.info(f"Added trusted agent: {agent_id}")
            
        except Exception as e:
            raise A2AEncryptionError(
                f"Failed to add trusted agent {agent_id}",
                context={"agent_id": agent_id, "error": str(e)}
            )
    
    def encrypt_message(self, message: A2AMessage, recipient_public_key: Optional[str] = None) -> Dict[str, Any]:
        """Encrypt A2A message"""
        try:
            # Get message content
            message_data = message.to_wire_format()
            message_json = json.dumps(message_data, sort_keys=True)
            message_bytes = message_json.encode('utf-8')
            
            if message.security_level == A2ASecurityLevel.NONE:
                return {
                    "encrypted": False,
                    "data": base64.b64encode(message_bytes).decode(),
                    "method": "none"
                }
            
            # Use symmetric encryption for performance
            if message.security_level in [A2ASecurityLevel.BASIC, A2ASecurityLevel.ENCRYPTED]:
                return self._encrypt_symmetric(message_bytes)
            
            # Use asymmetric encryption for higher security
            if recipient_public_key:
                return self._encrypt_asymmetric(message_bytes, recipient_public_key)
            elif message.to_agent in self.agent_keys:
                return self._encrypt_asymmetric(message_bytes, self.agent_keys[message.to_agent])
            else:
                # Fallback to symmetric encryption
                return self._encrypt_symmetric(message_bytes)
                
        except Exception as e:
            raise A2AEncryptionError(
                "Failed to encrypt message",
                context={
                    "message_id": message.message_id,
                    "from_agent": message.from_agent,
                    "to_agent": message.to_agent,
                    "error": str(e)
                }
            )
    
    def decrypt_message(self, encrypted_data: Dict[str, Any]) -> A2AMessage:
        """Decrypt A2A message"""
        try:
            if not encrypted_data.get("encrypted", False):
                # Message is not encrypted
                message_bytes = base64.b64decode(encrypted_data["data"])
                message_json = message_bytes.decode('utf-8')
                message_data = json.loads(message_json)
                return A2AMessage.parse_obj(message_data)
            
            method = encrypted_data.get("method", "symmetric")
            
            if method == "symmetric":
                message_bytes = self._decrypt_symmetric(encrypted_data)
            elif method == "asymmetric":
                message_bytes = self._decrypt_asymmetric(encrypted_data)
            else:
                raise A2AEncryptionError(f"Unknown encryption method: {method}")
            
            message_json = message_bytes.decode('utf-8')
            message_data = json.loads(message_json)
            return A2AMessage.parse_obj(message_data)
            
        except Exception as e:
            raise A2AEncryptionError(
                "Failed to decrypt message",
                context={"error": str(e)}
            )
    
    def _encrypt_symmetric(self, data: bytes) -> Dict[str, Any]:
        """Encrypt data using symmetric encryption"""
        try:
            # Generate random IV
            iv = os.urandom(16)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.symmetric_key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Pad data to block size
            padded_data = self._pad_data(data)
            
            # Encrypt
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return {
                "encrypted": True,
                "method": "symmetric",
                "data": base64.b64encode(ciphertext).decode(),
                "iv": base64.b64encode(iv).decode()
            }
            
        except Exception as e:
            raise A2AEncryptionError(f"Symmetric encryption failed: {str(e)}")
    
    def _decrypt_symmetric(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Decrypt data using symmetric encryption"""
        try:
            ciphertext = base64.b64decode(encrypted_data["data"])
            iv = base64.b64decode(encrypted_data["iv"])
            
            cipher = Cipher(
                algorithms.AES(self.symmetric_key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            return self._unpad_data(padded_data)
            
        except Exception as e:
            raise A2AEncryptionError(f"Symmetric decryption failed: {str(e)}")
    
    def _encrypt_asymmetric(self, data: bytes, recipient_key) -> Dict[str, Any]:
        """Encrypt data using asymmetric encryption"""
        try:
            # For large data, use hybrid encryption
            # Generate session key
            session_key = os.urandom(32)
            
            # Encrypt session key with recipient's public key
            if isinstance(recipient_key, str):
                recipient_key = serialization.load_pem_public_key(
                    recipient_key.encode(),
                    backend=self.backend
                )
            
            encrypted_session_key = recipient_key.encrypt(
                session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Encrypt data with session key
            iv = os.urandom(16)
            cipher = Cipher(
                algorithms.AES(session_key),
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            padded_data = self._pad_data(data)
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return {
                "encrypted": True,
                "method": "asymmetric",
                "data": base64.b64encode(ciphertext).decode(),
                "session_key": base64.b64encode(encrypted_session_key).decode(),
                "iv": base64.b64encode(iv).decode()
            }
            
        except Exception as e:
            raise A2AEncryptionError(f"Asymmetric encryption failed: {str(e)}")
    
    def _decrypt_asymmetric(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Decrypt data using asymmetric encryption"""
        try:
            encrypted_session_key = base64.b64decode(encrypted_data["session_key"])
            ciphertext = base64.b64decode(encrypted_data["data"])
            iv = base64.b64decode(encrypted_data["iv"])
            
            # Decrypt session key
            session_key = self.private_key.decrypt(
                encrypted_session_key,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            # Decrypt data with session key
            cipher = Cipher(
                algorithms.AES(session_key),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            return self._unpad_data(padded_data)
            
        except Exception as e:
            raise A2AEncryptionError(f"Asymmetric decryption failed: {str(e)}")
    
    def sign_message(self, message: A2AMessage) -> str:
        """Sign A2A message"""
        try:
            # Create signature payload
            signature_data = {
                "message_id": message.message_id,
                "from_agent": message.from_agent,
                "to_agent": message.to_agent,
                "message_type": message.message_type,
                "created_at": message.created_at.isoformat(),
                "payload_hash": self._hash_payload(message.payload.dict())
            }
            
            signature_json = json.dumps(signature_data, sort_keys=True)
            signature_bytes = signature_json.encode('utf-8')
            
            # Sign with private key
            signature = self.private_key.sign(
                signature_bytes,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return base64.b64encode(signature).decode()
            
        except Exception as e:
            raise A2ASignatureError(
                "Failed to sign message",
                context={
                    "message_id": message.message_id,
                    "from_agent": message.from_agent,
                    "error": str(e)
                }
            )
    
    def verify_signature(self, message: A2AMessage, signature: str, sender_public_key: Optional[str] = None) -> bool:
        """Verify message signature"""
        try:
            # Get sender's public key
            if sender_public_key:
                public_key = serialization.load_pem_public_key(
                    sender_public_key.encode(),
                    backend=self.backend
                )
            elif message.from_agent in self.agent_keys:
                public_key = self.agent_keys[message.from_agent]
            else:
                raise A2ASignatureError(f"No public key available for agent {message.from_agent}")
            
            # Recreate signature payload
            signature_data = {
                "message_id": message.message_id,
                "from_agent": message.from_agent,
                "to_agent": message.to_agent,
                "message_type": message.message_type,
                "created_at": message.created_at.isoformat(),
                "payload_hash": self._hash_payload(message.payload.dict())
            }
            
            signature_json = json.dumps(signature_data, sort_keys=True)
            signature_bytes = signature_json.encode('utf-8')
            
            # Verify signature
            signature_data = base64.b64decode(signature)
            
            try:
                public_key.verify(
                    signature_data,
                    signature_bytes,
                    padding.PSS(
                        mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.MAX_LENGTH
                    ),
                    hashes.SHA256()
                )
                return True
            except InvalidSignature:
                return False
                
        except Exception as e:
            raise A2ASignatureError(
                "Failed to verify signature",
                context={
                    "message_id": message.message_id,
                    "from_agent": message.from_agent,
                    "error": str(e)
                }
            )
    
    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        """Create hash of message payload"""
        payload_json = json.dumps(payload, sort_keys=True)
        payload_bytes = payload_json.encode('utf-8')
        hash_digest = hashlib.sha256(payload_bytes).digest()
        return base64.b64encode(hash_digest).decode()
    
    def _pad_data(self, data: bytes) -> bytes:
        """Pad data to block size for CBC mode"""
        block_size = 16
        padding_length = block_size - (len(data) % block_size)
        padding = bytes([padding_length] * padding_length)
        return data + padding
    
    def _unpad_data(self, padded_data: bytes) -> bytes:
        """Remove padding from data"""
        padding_length = padded_data[-1]
        return padded_data[:-padding_length]
    
    def generate_jwt_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """Generate JWT token for authentication"""
        try:
            payload.update({
                "iat": datetime.now(),
                "exp": datetime.now() + timedelta(seconds=expires_in),
                "iss": self.agent_id,
                "platform": "samsaek"
            })
            
            return jwt.encode(
                payload,
                self.get_private_key_pem(),
                algorithm="RS256"
            )
            
        except Exception as e:
            raise SecurityError(
                "Failed to generate JWT token",
                context={"agent_id": self.agent_id, "error": str(e)}
            )
    
    def verify_jwt_token(self, token: str, issuer_public_key: Optional[str] = None) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            if issuer_public_key:
                public_key = issuer_public_key
            else:
                # Extract issuer from token without verification
                unverified_payload = jwt.decode(token, options={"verify_signature": False})
                issuer = unverified_payload.get("iss")
                
                if issuer not in self.trusted_agents:
                    raise SecurityError(f"Unknown issuer: {issuer}")
                
                public_key = self.trusted_agents[issuer]
            
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"]
            )
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise SecurityError("JWT token has expired")
        except jwt.InvalidTokenError as e:
            raise SecurityError(f"Invalid JWT token: {str(e)}")
        except Exception as e:
            raise SecurityError(
                "Failed to verify JWT token",
                context={"error": str(e)}
            )