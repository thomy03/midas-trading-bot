#!/usr/bin/env python3
"""
Substack API Publisher
======================
Publie des articles sur Substack via l'API interne (pas Playwright).

Usage:
    python substack_api.py "Titre" "Contenu markdown"
    
Ou en Python:
    from substack_api import SubstackAPI
    api = SubstackAPI()
    url = api.publish("Titre", "Contenu", subtitle="Sous-titre")
"""

import requests
import json
import sys
import re
from pathlib import Path
from typing import Optional, List, Dict, Any

# Configuration
SUBSTACK_SUBDOMAIN = "aitradingradar"
BASE_URL = f"https://{SUBSTACK_SUBDOMAIN}.substack.com"
SESSION_FILE = Path("/root/substack_session.json")
USER_ID = 444618159  # AI Trading Radar author ID


class SubstackAPI:
    """Client API pour publier sur Substack."""
    
    def __init__(self, session_file: Path = SESSION_FILE):
        self.base_url = BASE_URL
        self.session = requests.Session()
        self._load_cookies(session_file)
    
    def _load_cookies(self, session_file: Path):
        """Charge les cookies depuis le fichier de session Playwright."""
        if not session_file.exists():
            raise FileNotFoundError(f"Session file not found: {session_file}")
        
        with open(session_file) as f:
            data = json.load(f)
        
        # Playwright storage state format
        for cookie in data.get("cookies", []):
            self.session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain", ".substack.com"),
                path=cookie.get("path", "/")
            )
    
    def markdown_to_prosemirror(self, markdown: str) -> Dict[str, Any]:
        """Convertit le Markdown en format ProseMirror pour Substack."""
        content = []
        lines = markdown.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Heading
            if line.startswith('# '):
                content.append({
                    "type": "heading",
                    "attrs": {"level": 1},
                    "content": [{"type": "text", "text": line[2:].strip()}]
                })
            elif line.startswith('## '):
                content.append({
                    "type": "heading",
                    "attrs": {"level": 2},
                    "content": [{"type": "text", "text": line[3:].strip()}]
                })
            elif line.startswith('### '):
                content.append({
                    "type": "heading",
                    "attrs": {"level": 3},
                    "content": [{"type": "text", "text": line[4:].strip()}]
                })
            # Horizontal rule
            elif line.strip() in ['---', '***', '___']:
                content.append({"type": "horizontal_rule"})
            # Bullet list item
            elif line.strip().startswith('- ') or line.strip().startswith('* '):
                # Collect all list items
                list_items = []
                while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('* ')):
                    item_text = lines[i].strip()[2:]
                    list_items.append({
                        "type": "list_item",
                        "content": [{"type": "paragraph", "content": self._parse_inline(item_text)}]
                    })
                    i += 1
                content.append({
                    "type": "bullet_list",
                    "content": list_items
                })
                continue
            # Empty line
            elif not line.strip():
                pass
            # Regular paragraph
            else:
                if line.strip():
                    content.append({
                        "type": "paragraph",
                        "content": self._parse_inline(line)
                    })
            
            i += 1
        
        return {"type": "doc", "content": content}
    
    def _parse_inline(self, text: str) -> List[Dict]:
        """Parse inline formatting (bold, italic, links)."""
        result = []
        
        # Simple parsing - just return text for now
        # Could be enhanced to handle **bold**, *italic*, [links](url)
        if text.strip():
            # Handle bold
            parts = re.split(r'\*\*(.*?)\*\*', text)
            for j, part in enumerate(parts):
                if not part:
                    continue
                if j % 2 == 1:  # Bold
                    result.append({
                        "type": "text",
                        "text": part,
                        "marks": [{"type": "strong"}]
                    })
                else:
                    result.append({"type": "text", "text": part})
            
            if not result:
                result.append({"type": "text", "text": text})
        
        return result if result else [{"type": "text", "text": " "}]
    
    def create_draft(self, title: str, content: str, subtitle: str = "", audience: str = "everyone") -> Optional[int]:
        """Crée un brouillon et retourne son ID."""
        body = self.markdown_to_prosemirror(content)
        
        payload = {
            "draft_title": title,
            "draft_subtitle": subtitle,
            "draft_body": json.dumps(body),
            "draft_bylines": [{"id": USER_ID, "is_guest": False}],
            "audience": audience,  # "everyone" ou "only_paid"
            "type": "newsletter"
        }
        
        resp = self.session.post(
            f"{self.base_url}/api/v1/drafts",
            json=payload
        )
        
        if resp.status_code == 200:
            data = resp.json()
            draft_id = data.get("id")
            print(f"✅ Draft créé: ID {draft_id}")
            return draft_id
        else:
            print(f"❌ Erreur création draft: {resp.status_code}")
            print(resp.text[:500])
            return None
    
    def prepublish(self, draft_id: int) -> bool:
        """Prépare le draft pour publication."""
        resp = self.session.get(f"{self.base_url}/api/v1/drafts/{draft_id}/prepublish")
        
        if resp.status_code == 200:
            print(f"✅ Prepublish OK")
            return True
        else:
            print(f"⚠️ Prepublish: {resp.status_code} (peut être ignoré)")
            return True  # Parfois retourne 4xx mais fonctionne quand même
    
    def publish(self, title: str, content: str, subtitle: str = "", audience: str = "everyone") -> Optional[str]:
        """
        Publie un article complet sur Substack.
        
        Args:
            title: Titre de l'article
            content: Contenu en Markdown
            subtitle: Sous-titre (optionnel)
            audience: "everyone" ou "only_paid"
            
        Returns:
            URL de l'article publié ou None
        """
        # 1. Créer le draft
        draft_id = self.create_draft(title, content, subtitle, audience)
        if not draft_id:
            return None
        
        # 2. Prepublish
        self.prepublish(draft_id)
        
        # 3. Publish
        resp = self.session.post(
            f"{self.base_url}/api/v1/drafts/{draft_id}/publish",
            json={"send": True}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            slug = data.get("slug", "")
            post_id = data.get("id", draft_id)
            url = f"{self.base_url}/p/{slug}" if slug else f"{self.base_url}/p/{post_id}"
            print(f"✅ Publié: {url}")
            return url
        else:
            print(f"❌ Erreur publication: {resp.status_code}")
            print(resp.text[:500])
            return None


def publish_report(title: str, content: str, subtitle: str = "") -> Optional[str]:
    """Fonction utilitaire pour publier rapidement."""
    api = SubstackAPI()
    return api.publish(title, content, subtitle)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python substack_api.py 'Titre' 'Contenu markdown'")
        print("       python substack_api.py 'Titre' 'Contenu' 'Sous-titre'")
        sys.exit(1)
    
    title = sys.argv[1]
    content = sys.argv[2]
    subtitle = sys.argv[3] if len(sys.argv) > 3 else ""
    
    url = publish_report(title, content, subtitle)
    if url:
        print(f"\n🎉 Article publié: {url}")
    else:
        print("\n❌ Échec de la publication")
        sys.exit(1)
