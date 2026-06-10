"""Runtime compatibility patches loaded automatically by Python site initialization.

This file is placed on VERL_PATH, which is prepended to PYTHONPATH by the
training scripts. Python imports ``sitecustomize`` automatically in both the
main process and multiprocessing-spawn child processes, so compatibility patches
here also apply to SGLang scheduler/server subprocesses.
"""


def _patch_transformers_auto_image_processor_register_for_sglang():
    """Patch transformers compatibility for older SGLang image processor registration.

    Some SGLang versions call::

        AutoImageProcessor.register(config, None, image_processor, None)

    With newer transformers versions, the third positional argument is treated as
    ``fast_image_processor_class`` and must inherit from ``BaseImageProcessorFast``.
    For SGLang's old call shape, remap it to the compatible slow image processor
    registration form.
    """
    try:
        from transformers import AutoImageProcessor
    except Exception:
        return

    original_register = AutoImageProcessor.register
    if getattr(original_register, "_verl_sglang_compat_patched", False):
        return

    def register_compat(config_class, *args, **kwargs):
        exist_ok = kwargs.pop("exist_ok", True)
        try:
            return original_register(config_class, *args, exist_ok=exist_ok, **kwargs)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            is_sglang_old_signature = (
                len(args) >= 3
                and args[0] is None
                and args[1] is not None
                and args[2] is None
                and (
                    "multiple values for argument 'exist_ok'" in message
                    or "fast_image_processor_class" in message
                )
            )
            if not is_sglang_old_signature:
                raise
            return original_register(config_class, args[1], exist_ok=exist_ok, **kwargs)

    register_compat._verl_sglang_compat_patched = True
    AutoImageProcessor.register = register_compat


_patch_transformers_auto_image_processor_register_for_sglang()
