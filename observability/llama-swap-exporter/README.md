# Llama-swap Prometheus Exporter

Экспортер получает список запущенных моделей в llama-swap, обращается к ним через `/upstream/{model_name}/metrics`, заменяет префикс `llamacpp:` на `llamacpp_` и добавляет лейбл с именем модели. Предполагается, что модели запускаются с помощью llama.cpp.

Также экспортер получает json метрики llama-swap, которые отображаются на странице Activity, через `/api/metrics/activity`, оставляет последний элемент для каждой модели, преобразует в формат Prometheus.

## Эндпоинты

- `GET /metrics` — метрики Prometheus (порт по умолчанию `8081`)
- `GET /health` — проверка доступности (возвращает `OK`)

## Метрики

1. llamacpp_prompt_tokens_total
2. llamacpp_prompt_tokens_cached_total
3. llamacpp_prompt_seconds_total
4. llamacpp_tokens_predicted_total
5. llamacpp_tokens_predicted_seconds_total
6. llamacpp_n_decode_total
7. llamacpp_n_tokens_max_total
8. llamacpp_spec_decode_num_draft_tokens_total
9. llamacpp_spec_decode_num_accepted_tokens_total
10. llamacpp_spec_decode_num_drafts_total
11. llamacpp_spec_decode_num_accepted_tokens_per_pos_total
12. llamacpp_prompt_tokens_seconds
13. llamacpp_predicted_tokens_seconds
14. llamacpp_requests_processing
15. llamacpp_requests_deferred
16. llamacpp_n_busy_slots_per_decode
17. llamaswap_model_cache_tokens
18. llamaswap_model_input_tokens
19. llamaswap_model_output_tokens
20. llamaswap_model_prompt_per_second
21. llamaswap_model_tokens_per_second
22. llamaswap_model_duration_ms
23. llamaswap_model_draft_tokens
24. llamaswap_model_draft_acc_tokens
