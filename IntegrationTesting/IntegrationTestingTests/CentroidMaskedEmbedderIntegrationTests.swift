// Copyright © 2026 Joel Nishanth / Offlyn AI. Apache-2.0 license.

import Foundation
import IntegrationTestHelpers
import MLX
import MLXLMCommon
import MLXNN
@_spi(Testing) import MLXVLM
import Testing

private func hfSnapshotDir(modelId: String) -> URL? {
    let home = FileManager.default.homeDirectoryForCurrentUser
    let hub = home.appendingPathComponent(".cache/huggingface/hub")
    let folderName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
    let snapshots = hub.appendingPathComponent(folderName).appendingPathComponent("snapshots")
    guard
        let entries = try? FileManager.default.contentsOfDirectory(
            at: snapshots, includingPropertiesForKeys: nil)
    else { return nil }
    return entries.first
}

private func localCentroidFixtureURL() -> URL? {
    let repoRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
    let fixture = repoRoot
        .appendingPathComponent("tools/fixtures/centroid_masked_embedder/case_01.safetensors")
    return FileManager.default.fileExists(atPath: fixture.path) ? fixture : nil
}

@Suite(.serialized)
struct CentroidMaskedEmbedderIntegrationTests {

    @Test
    func testE2BMaskedEmbedderMatchesPythonFixture() async throws {
        guard let fixtureURL = localCentroidFixtureURL() else {
            Issue.record(
                "centroid fixture missing — run tools/generate_centroid_fixtures.py first")
            return
        }
        guard
            let drafterDir = hfSnapshotDir(
                modelId: "mlx-community/gemma-4-E2B-it-assistant-bf16")
        else {
            Issue.record("E2B assistant checkpoint not in HF cache; skipping centroid test")
            return
        }

        let configURL = drafterDir.appendingPathComponent("config.json")
        let cfg = try JSONDecoder().decode(
            Gemma4AssistantConfiguration.self, from: Data(contentsOf: configURL))
        #expect(cfg.useOrderedEmbeddings, "E2B assistant must use ordered embeddings")

        let model = Gemma4AssistantDraftModel(cfg)
        try loadWeights(modelDirectory: drafterDir, model: model)

        guard let masked = model.maskedEmbedding else {
            Issue.record("maskedEmbedding module not initialized despite use_ordered_embeddings=true")
            return
        }

        let arrays = try MLX.loadArrays(url: fixtureURL)
        guard
            let hiddenStates = arrays["inputs/hidden_states"],
            let expectedLogits = arrays["outputs/logits"]
        else {
            Issue.record("centroid fixture missing expected tensor keys")
            return
        }

        let lmHeadWeight = model.model.embedTokens.weight
        let logits = masked(hiddenStates, lmHeadWeight: lmHeadWeight)

        #expect(logits.shape == expectedLogits.shape)
        #expect(!isNaN(logits).any().item(Bool.self), "Swift logits contain NaN")
        #expect(!isInf(logits).any().item(Bool.self), "Swift logits contain Inf")

        let swiftF32 = logits.asType(.float32)
        let expectedF32 = expectedLogits.asType(.float32)
        #expect(
            allClose(swiftF32, expectedF32, rtol: 1e-3, atol: 1e-3).item(Bool.self),
            "centroid masked logits diverge from Python fixture")
    }
}
