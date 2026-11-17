"""编辑领域仓库接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from app.schemas.edit import (
    EditTaskRecord,
    GenerationRecord,
    GenerationStepRecord,
    PersonRecord,
    RegionRecord,
    StoredImage,
)


class EditRepository(ABC):
    @abstractmethod
    def save_image(self, image: StoredImage) -> None: ...

    @abstractmethod
    def get_image(self, image_id: str) -> StoredImage | None: ...

    @abstractmethod
    def update_image(self, image: StoredImage) -> None: ...

    @abstractmethod
    def save_person(self, person: PersonRecord) -> None: ...

    @abstractmethod
    def get_person(self, person_id: str) -> PersonRecord | None: ...

    @abstractmethod
    def list_persons(self, image_id: str) -> list[PersonRecord]: ...

    @abstractmethod
    def save_region(self, region: RegionRecord) -> None: ...

    @abstractmethod
    def get_region(self, region_id: str) -> RegionRecord | None: ...

    @abstractmethod
    def list_regions(self, person_id: str) -> list[RegionRecord]: ...

    @abstractmethod
    def list_regions_by_image(self, image_id: str) -> list[RegionRecord]: ...

    @abstractmethod
    def save_task(self, task: EditTaskRecord) -> None: ...

    @abstractmethod
    def get_task(self, task_id: str) -> EditTaskRecord | None: ...

    @abstractmethod
    def list_tasks(
        self,
        *,
        image_id: str | None = None,
        person_id: str | None = None,
    ) -> list[EditTaskRecord]: ...

    @abstractmethod
    def get_task_for_region(
        self,
        person_id: str,
        region_id: str,
    ) -> EditTaskRecord | None: ...

    @abstractmethod
    def delete_task(self, task_id: str) -> None: ...

    @abstractmethod
    def save_generation(self, generation: GenerationRecord) -> None: ...

    @abstractmethod
    def get_generation(self, generation_id: str) -> GenerationRecord | None: ...

    @abstractmethod
    def find_active_generation(
        self,
        *,
        person_id: str,
        plan_hash: str,
    ) -> GenerationRecord | None: ...

    @abstractmethod
    def find_generation_by_idempotency(
        self,
        idempotency_key: str,
    ) -> GenerationRecord | None: ...

    @abstractmethod
    def list_active_generations(self, person_id: str) -> list[GenerationRecord]: ...

    @abstractmethod
    def save_step(self, step: GenerationStepRecord) -> None: ...

    @abstractmethod
    def update_step(self, step: GenerationStepRecord) -> None: ...

    @abstractmethod
    def list_steps(self, generation_id: str) -> list[GenerationStepRecord]: ...

    @abstractmethod
    def mark_running_generations_interrupted(self) -> int: ...

    @abstractmethod
    def list_expired_image_ids(self, now: datetime) -> list[str]: ...

    @abstractmethod
    def list_generation_ids(self, image_id: str) -> list[str]: ...

    @abstractmethod
    def delete_image_data(self, image_id: str) -> None: ...
