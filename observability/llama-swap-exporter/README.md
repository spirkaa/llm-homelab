# Llama-swap Prometheus Exporter

Экспортер получает список запущенных моделей в llama-swap, обращается к ним через `/upstream/{model_name}/metrics`, заменяет префикс `llamacpp:` на `llamacpp_` и добавляет лейбл с именем модели. Предполагается, что модели запускаются с помощью llama.cpp.

Также экспортер получает json метрики llama-swap, которые отображаются на странице Activity, через `/api/metrics`, оставляет последний элемент для каждой модели, преобразует в формат Prometheus.

## Метрики

1. llamacpp_prompt_tokens_total
2. llamacpp_prompt_seconds_total
3. llamacpp_tokens_predicted_total
4. llamacpp_tokens_predicted_seconds_total
5. llamacpp_n_decode_total
6. llamacpp_n_tokens_max_total
7. llamacpp_prompt_tokens_seconds
8. llamacpp_predicted_tokens_seconds
9. llamacpp_requests_processing
10. llamacpp_requests_deferred
11. llamacpp_n_busy_slots_per_decode
12. llamaswap_model_cache_tokens
13. llamaswap_model_input_tokens
14. llamaswap_model_output_tokens
15. llamaswap_model_prompt_per_second
16. llamaswap_model_tokens_per_second
17. llamaswap_model_duration_ms
