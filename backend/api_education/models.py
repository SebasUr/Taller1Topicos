from django.db import models
from django.conf import settings

class Module(models.Model):
    title = models.CharField(max_length=255)
    description = models.TextField()

    def __str__(self):
        return self.title


class Lesson(models.Model):
    module = models.ForeignKey(Module, on_delete=models.CASCADE, related_name="lessons")
    title = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.title} - {self.module.title}"

    @property
    def content(self):
        return list(
            self.content_blocks.order_by("order").values("block_type", "data")
        )


class LessonContentBlock(models.Model):
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE, related_name="content_blocks")
    order = models.PositiveIntegerField()
    block_type = models.CharField(max_length=50)  # e.g., 'text', 'image', etc.
    data = models.JSONField()

    class Meta:
        ordering = ['order']

    def __str__(self):
        return f"{self.lesson.title} - Block {self.order} ({self.block_type})"


class UserProgress(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    lesson = models.ForeignKey(Lesson, on_delete=models.CASCADE)
    is_completed = models.BooleanField(default=False)

    class Meta:
        unique_together = ('user', 'lesson')

    def __str__(self):
        return f"{self.user.username} - {self.lesson.title}: {self.is_completed}"
