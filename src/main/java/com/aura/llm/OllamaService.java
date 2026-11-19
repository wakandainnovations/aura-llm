package com.aura.llm;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.HashMap;
import java.util.Map;

/**
 * Calls Ollama's /api/generate with the primary model ({@code ollama.model}, the rate-limited
 * "-cloud" model). When that returns HTTP 429, the same prompt is retried once on a local fallback
 * model ({@code ollama.fallback.model}, e.g. "mistral"), and for {@code ollama.fallback.cooldown.seconds}
 * after a 429 every call goes straight to the fallback so the cloud model isn't hammered while it is
 * limiting us. Leave {@code ollama.fallback.model} empty to disable the fallback.
 */
@Service
public class OllamaService implements LlmService {
    @Value("${ollama.api.url}")
    private String ollamaUrl;

    @Value("${ollama.model}")
    private String modelName;

    @Value("${ollama.fallback.model:}")
    private String fallbackModel;

    // Ollama's default context window (2048-4096 tokens) is smaller than the longest sentiment
    // prompts (~3k tokens before the post text), and Ollama truncates silently - so set it explicitly.
    @Value("${ollama.fallback.num_ctx:8192}")
    private int fallbackNumCtx;

    // A local model on this Mac is much slower than the cloud one. Kept under the sentiment
    // analyser's 120s socket timeout so it gets an answer (or a clear error) before giving up.
    @Value("${ollama.fallback.timeout.seconds:110}")
    private long fallbackTimeoutSeconds;

    @Value("${ollama.fallback.cooldown.seconds:60}")
    private long fallbackCooldownSeconds;

    private volatile long primaryRateLimitedUntilMillis = 0;

    private final ObjectMapper objectMapper = new ObjectMapper();
    private final HttpClient client = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_2)
            .connectTimeout(Duration.ofSeconds(10))
            .build();

    // Adapts the existing Ollama call to the common LlmService contract.
    @Override
    public String generate(String prompt) {
        return callOllama(prompt);
    }

    public String callOllama(String promptText) {
        boolean fallbackEnabled = fallbackModel != null && !fallbackModel.isBlank();
        if (fallbackEnabled && System.currentTimeMillis() < primaryRateLimitedUntilMillis) {
            return callModel(fallbackModel.trim(), promptText, Duration.ofSeconds(fallbackTimeoutSeconds), fallbackNumCtx);
        }

        // Ollama's "-cloud" models proxy to remote infra and can have a slow cold start;
        // bound the wait so a stalled call fails fast instead of hanging indefinitely.
        String result = callModel(modelName, promptText, Duration.ofSeconds(55), null);
        if (fallbackEnabled && result.equals(errorFor(429))) {
            primaryRateLimitedUntilMillis = System.currentTimeMillis() + fallbackCooldownSeconds * 1000;
            System.err.println("Ollama model '" + modelName + "' returned 429; using local fallback '"
                    + fallbackModel.trim() + "' for the next " + fallbackCooldownSeconds + "s.");
            return callModel(fallbackModel.trim(), promptText, Duration.ofSeconds(fallbackTimeoutSeconds), fallbackNumCtx);
        }
        return result;
    }

    private String callModel(String model, String promptText, Duration timeout, Integer numCtx) {
        try {
            Map<String, Object> jsonMap = new HashMap<>();
            jsonMap.put("model", model);
            jsonMap.put("prompt", promptText);
            jsonMap.put("stream", false);
            if (numCtx != null) {
                jsonMap.put("options", Map.of("num_ctx", numCtx));
            }

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(ollamaUrl))
                    .header("Content-Type", "application/json")
                    .timeout(timeout)
                    .POST(HttpRequest.BodyPublishers.ofString(objectMapper.writeValueAsString(jsonMap)))
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() == 200) {
                JsonNode rootNode = objectMapper.readTree(response.body());
                return rootNode.path("response").asText();
            }
            return errorFor(response.statusCode());

        } catch (Exception e) {
            e.printStackTrace();
            return "Internal Server Error: " + e.getMessage();
        }
    }

    private static String errorFor(int statusCode) {
        return "Error from Ollama: " + statusCode;
    }
}
