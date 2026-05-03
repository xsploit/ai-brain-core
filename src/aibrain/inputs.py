from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel


class ImageInput(BaseModel):
    path: str | None = None
    url: str | None = None
    file_id: str | None = None
    base64_data: str | None = None
    mime_type: str | None = None
    detail: Literal["low", "high", "auto"] = "auto"

    def to_content(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "input_image", "detail": self.detail}
        if self.file_id:
            payload["file_id"] = self.file_id
            return payload
        if self.url:
            payload["image_url"] = self.url
            return payload
        if self.path:
            path = Path(self.path)
            mime = self.mime_type or mimetypes.guess_type(path.name)[0] or "image/png"
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            payload["image_url"] = f"data:{mime};base64,{data}"
            return payload
        if self.base64_data:
            mime = self.mime_type or "image/png"
            payload["image_url"] = f"data:{mime};base64,{self.base64_data}"
            return payload
        raise ValueError("ImageInput needs path, url, file_id, or base64_data")


class FileInput(BaseModel):
    path: str | None = None
    file_id: str | None = None
    file_url: str | None = None
    filename: str | None = None

    def to_content(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": "input_file"}
        if self.file_id:
            payload["file_id"] = self.file_id
            return payload
        if self.file_url:
            payload["file_url"] = self.file_url
            if self.filename:
                payload["filename"] = self.filename
            return payload
        if self.path:
            path = Path(self.path)
            payload["filename"] = self.filename or path.name
            payload["file_data"] = base64.b64encode(path.read_bytes()).decode("ascii")
            return payload
        raise ValueError("FileInput needs path, file_id, or file_url")


def normalize_image(value: ImageInput | dict[str, Any] | str) -> ImageInput:
    if isinstance(value, ImageInput):
        return value
    if isinstance(value, dict):
        return ImageInput.model_validate(value)
    if value.startswith(("http://", "https://", "data:")):
        return ImageInput(url=value)
    return ImageInput(path=value)


def normalize_file(value: FileInput | dict[str, Any] | str) -> FileInput:
    if isinstance(value, FileInput):
        return value
    if isinstance(value, dict):
        return FileInput.model_validate(value)
    if value.startswith(("http://", "https://")):
        return FileInput(file_url=value)
    return FileInput(path=value)


def build_user_message(
    text: str,
    *,
    images: list[ImageInput | dict[str, Any] | str] | None = None,
    files: list[FileInput | dict[str, Any] | str] | None = None,
) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for image in images or []:
        content.append(normalize_image(image).to_content())
    for file in files or []:
        content.append(normalize_file(file).to_content())
    return {"type": "message", "role": "user", "content": content}
