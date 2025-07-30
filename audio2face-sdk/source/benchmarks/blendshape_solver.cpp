// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
#include "utils.h"

#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include "audio2x/cuda_stream.h"

#include <benchmark/benchmark.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "cnpy.h"

#include <future>
#include <random>
#include <string_view>
#include <unordered_map>


namespace {

enum class GeometryType {
    UNKNOWN,
    SKIN,
    TONGUE,
};

std::unordered_map<GeometryType, std::string> Geometry2NpzKey = {
    {GeometryType::UNKNOWN, "unknown"},
    {GeometryType::SKIN,    "inferred_poses"},
    {GeometryType::TONGUE,  "inferred_tongue_poses"},
};

struct TestParam {
    std::string_view identity{};
    GeometryType geometryType{GeometryType::UNKNOWN};
    std::string_view bsDataPath{};
};

const std::vector<TestParam> testParams = {
    // skin
    {
        "Claire",
        GeometryType::SKIN,
        TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/claire/bs_skin.npz",
    },
    {
        "James",
        GeometryType::SKIN,
        TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/james/bs_skin.npz",
    },
    {
        "Mark",
        GeometryType::SKIN,
        TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/bs_skin.npz",
    },
    // tongue
    {
        "Claire",
        GeometryType::TONGUE,
        TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/claire/bs_tongue.npz",
    },
    {
        "James",
        GeometryType::TONGUE,
        TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/james/bs_tongue.npz",
    },
    {
        "Mark",
        GeometryType::TONGUE,
        TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/mark/bs_tongue.npz",
    },
};


std::vector<float> generateTargetPose(const UniquePtr<nva2f::IBlendshapeSolver>& solver) {
    std::vector<float> targetPose(solver->PoseSize());
    std::vector<float> weights(solver->NumBlendshapePoses());

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);  // Range [0,1)

    // most values near 0, some activated up to 1.0, using squared uniform distribution
    for (int i = 0; i < weights.size(); ++i) {
        float val = dist(rng);
        weights[i] = val * val;
    }

    if (solver->EvaluatePose(
        nva2x::HostTensorFloatConstView{weights.data(), weights.size()}, 
        nva2x::HostTensorFloatView{targetPose.data(), targetPose.size()}
    )) {
        return {};
    }
    return targetPose;
}

} // Anonymouse namespace

class BlendshapeSolverBenchmark : public benchmark::Fixture {
public:
    UniquePtr<nva2x::ICudaStream> cudaStream;
    UniquePtr<nva2f::IJobRunner> jobRunner;
    UniquePtr<nva2f::IBlendshapeSolver> solver;
    std::vector<float> targetPose;

    bool init{false};
    TestParam testParam;

    void SetUp(const ::benchmark::State& state) override {
        init = false;
        bool useGPUSolver = state.range(0);
        testParam = testParams[state.range(1)];
        if (nva2x::SetCudaDeviceIfNeeded(kDeviceID)) {
            std::cout << "SetCudaDeviceIfNeeded failed." << std::endl;
            return;
        }
        cudaStream.reset(nva2x::CreateCudaStream());
        jobRunner.reset(nva2f::CreateThreadPoolJobRunner(1));
        solver.reset(nva2f::CreateBlendshapeSolver(useGPUSolver));

        solver->SetCudaStream(cudaStream->Data());
        solver->SetJobRunner(jobRunner.get());

        auto solverData = ToUniquePtr(
            nva2f::ReadBlendshapeSolverData(testParam.bsDataPath.data())
        );
        if (!solverData) {
            std::cout << "ReadBlendshapeSolverData failed." << std::endl;
            return;
        }
        if (solver->SetBlendshapeData(solverData->GetBlendshapeSolverDataView())) {
            std::cout << "SetBlendshapeData failed." << std::endl;
            return;
        }

        if (solver->Prepare()) {
            std::cout << "Prepare failed." << std::endl;
            return;
        }

        targetPose = generateTargetPose(solver);
        if (targetPose.empty()) {
            std::cout << "Generate targetPose failed." << std::endl;
            return;
        }
        if (targetPose.size() % solver->PoseSize() != 0) {
            std::cout << "Size mismatch." << std::endl;
            return;
        }
        init = true;
    }

    void TearDown(const ::benchmark::State& state) override {

    }
};

BENCHMARK_DEFINE_F(BlendshapeSolverBenchmark, BM_Solve)(benchmark::State& state) {
    if (!init) {
        state.SkipWithError("BlendshapeSolverBenchmark::SetUp() failed.");
    }
    std::ostringstream label;
    label << "useGPUSolver: " << (state.range(0) ? "true" : "false")
        << ", identity: " << testParam.identity
        << ", geometry: " << Geometry2NpzKey[testParam.geometryType];
    state.SetLabel(label.str());

    const size_t poseSize = solver->PoseSize();
    const size_t nbFrames = targetPose.size() / poseSize;
    auto targetPoseDevice = ToUniquePtr(nva2x::CreateDeviceTensorFloat({targetPose.data(), targetPose.size()}, cudaStream->Data()));
    auto solvedWeights = ToUniquePtr(nva2x::CreateHostTensorFloat(solver->NumBlendshapePoses()));

    // warm-up
    for(int i=0;i<nbFrames;++i) {
        if (solver->Solve(targetPoseDevice->View(i * poseSize, poseSize), *solvedWeights)) {
            state.SkipWithError("Warmup failed.");
        }
    }

    int idx = 0;
    for (auto _ : state) {
        if (solver->Solve(targetPoseDevice->View(idx * poseSize, poseSize), *solvedWeights)) {
            state.SkipWithError("Solve failed.");
        }
        idx = (idx + 1) % nbFrames;
    }
}
BENCHMARK_REGISTER_F(BlendshapeSolverBenchmark, BM_Solve)->Apply([](benchmark::internal::Benchmark* b){
    b->UseRealTime(); // Configures the benchmark to use real-time (wall clock) instead of CPU time.
                  // This is necessary for the CPU solver, which operates in a multi-threaded environment,
                  // as CPU time measurement may not accurately reflect execution duration when multiple
                  // threads are involved.
    for(int i : {false, true}) {
        for(int j=0;j<testParams.size();++j) {
            b->Args({i, j});
        }
    }
});

BENCHMARK_DEFINE_F(BlendshapeSolverBenchmark, BM_CPUSolveAsync)(benchmark::State& state) {
    if (!init) {
        state.SkipWithError("BlendshapeSolverBenchmark::SetUp() failed.");
    }
    std::ostringstream label;
    label << "useGPUSolver: " << (state.range(0) ? "true" : "false")
          << ", identity: " << testParam.identity
          << ", geometry: " << Geometry2NpzKey[testParam.geometryType];
    state.SetLabel(label.str());

    const auto numAsyncCalls = state.range(2);

    const size_t poseSize = solver->PoseSize();
    const size_t nbFrames = targetPose.size() / poseSize;
    auto targetPoseDevice = ToUniquePtr(nva2x::CreateDeviceTensorFloat({targetPose.data(), targetPose.size()}, cudaStream->Data()));
    std::vector<UniquePtr<nva2x::IHostTensorFloat>> solvedWeightsBatch;
    for(int i=0;i<numAsyncCalls;++i) {
        solvedWeightsBatch.emplace_back(nva2x::CreateHostTensorFloat(solver->NumBlendshapePoses()));
    }

    // warm-up
    {
        int solvedWeightsIdx = 0;
        for(int i=0;i<nbFrames;++i) {
            if (solver->Solve(targetPoseDevice->View(i * poseSize, poseSize), *(solvedWeightsBatch[solvedWeightsIdx]))) {
                state.SkipWithError("Warmup failed.");
            }
            solvedWeightsIdx = (solvedWeightsIdx + 1) % solvedWeightsBatch.size();
        }
    }

    int idx = 0;
    for (auto _ : state) {
        std::vector<std::promise<std::error_code>> promises(numAsyncCalls);
        std::vector<std::future<std::error_code>> futures;
        for(int i=0;i<numAsyncCalls;++i) {
            futures.emplace_back(promises[i].get_future());
        }

        // Run SolveAsync consecutively for numAsyncCalls times
        for(int i=0;i<numAsyncCalls;++i) {
            if (solver->SolveAsync(targetPoseDevice->View(idx * poseSize, poseSize), *(solvedWeightsBatch[i]), [](void* p, std::error_code error) {
                reinterpret_cast<std::promise<std::error_code>*>(p)->set_value(error);
            }, &promises[i])) {
                state.SkipWithError("SolveAsync failed.");
            }
            idx = (idx + 1) % nbFrames;
        }
        // Wait
        for (auto& future : futures) {
            if (future.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
                state.SkipWithError("Blendshape solve did not complete in 3 seconds");
            }
            if (future.get()) {
                state.SkipWithError("SolveAsync callback returned error.");
            }
        }
    }
    state.SetItemsProcessed(numAsyncCalls * state.iterations()); // Ensure the time reported is still for "one" SolveAsync
}
BENCHMARK_REGISTER_F(BlendshapeSolverBenchmark, BM_CPUSolveAsync)->Apply([](benchmark::internal::Benchmark* b){
    b->UseRealTime(); // Configures the benchmark to use real-time (wall clock) instead of CPU time.
                  // This is necessary for the CPU solver, which operates in a multi-threaded environment,
                  // as CPU time measurement may not accurately reflect execution duration when multiple
                  // threads are involved.
    for(int numCalls : {1, 2, 5, 10}) {
        for(int j=0;j<testParams.size();++j) {
            b->Args({false, j, numCalls});
        }
    }
});

BENCHMARK_DEFINE_F(BlendshapeSolverBenchmark, BM_GPUSolveAsync)(benchmark::State& state) {
    if (!init) {
        state.SkipWithError("BlendshapeSolverBenchmark::SetUp() failed.");
    }
    std::ostringstream label;
    label << "useGPUSolver: " << (state.range(0) ? "true" : "false")
          << ", identity: " << testParam.identity
          << ", geometry: " << Geometry2NpzKey[testParam.geometryType];
    state.SetLabel(label.str());

    const auto numAsyncCalls = state.range(2);

    const size_t poseSize = solver->PoseSize();
    const size_t nbFrames = targetPose.size() / poseSize;
    auto targetPoseDevice = ToUniquePtr(nva2x::CreateDeviceTensorFloat({targetPose.data(), targetPose.size()}, cudaStream->Data()));
    std::vector<UniquePtr<nva2x::IDeviceTensorFloat>> solvedWeightsDeviceBatch;
    std::vector<UniquePtr<nva2x::IHostTensorFloat>> solvedWeightsHostBatch;
    for(int i=0;i<numAsyncCalls;++i) {
        solvedWeightsDeviceBatch.emplace_back(nva2x::CreateDeviceTensorFloat(solver->NumBlendshapePoses()));
        solvedWeightsHostBatch.emplace_back(nva2x::CreateHostPinnedTensorFloat(solver->NumBlendshapePoses()));
    }

    // warm-up
    {
        int solvedWeightsIdx = 0;
        for(int i=0;i<nbFrames;++i) {
            if (solver->Solve(targetPoseDevice->View(i * poseSize, poseSize), *(solvedWeightsHostBatch[solvedWeightsIdx]))) {
                state.SkipWithError("Warmup failed.");
            }
            solvedWeightsIdx = (solvedWeightsIdx + 1) % solvedWeightsHostBatch.size();
        }
    }

    int idx = 0;
    for (auto _ : state) {
        // Run SolveAsync consecutively for numAsyncCalls times
        for(int i=0;i<numAsyncCalls;++i) {
            if (solver->SolveAsync(targetPoseDevice->View(idx * poseSize, poseSize), *solvedWeightsDeviceBatch[i])) {
                state.SkipWithError("SolveAsync failed.");
            }
            // Ensure the output is to a host buffer for a fair comparison with BM_CPUSolveAsync
            if (nva2x::CopyDeviceToHost(*solvedWeightsHostBatch[i], *solvedWeightsDeviceBatch[i], cudaStream->Data())) {
                state.SkipWithError("CopyDeviceToHost failed.");
            }
        }
        // Wait
        cudaStream->Synchronize();
    }
    state.SetItemsProcessed(numAsyncCalls * state.iterations()); // Ensure the time reported is still for "one" SolveAsync
}
BENCHMARK_REGISTER_F(BlendshapeSolverBenchmark, BM_GPUSolveAsync)->Apply([](benchmark::internal::Benchmark* b){
    b->UseRealTime();
    for(int numCalls : {1, 2, 5, 10}) {
        for(int j=0;j<testParams.size();++j) {
            b->Args({true, j, numCalls});
        }
    }
});

class BlendshapeSolverBatchBenchmark : public benchmark::Fixture {
public:
    size_t batchSize;
    TestParam testParam;

    void SetUp(const ::benchmark::State& state) override {
        batchSize = state.range(0);
        testParam = testParams[state.range(1)];
    }

    void TearDown(const ::benchmark::State& state) override {

    }
};

BENCHMARK_DEFINE_F(BlendshapeSolverBatchBenchmark, BM_CPUSolveAsync)(benchmark::State& state) {
    auto jobRunner = ToUniquePtr(nva2f::CreateThreadPoolJobRunner(batchSize));
    if (!jobRunner) {
        return;
    }
    auto solverData = ToUniquePtr(nva2f::ReadBlendshapeSolverData(testParam.bsDataPath.data()));
    if (!solverData) {
        std::cout << "ReadBlendshapeSolverData failed." << std::endl;
        return;
    }
    std::vector<UniquePtr<nva2x::ICudaStream>> cudaStreams;
    std::vector<UniquePtr<nva2f::IBlendshapeSolver>> solvers;
    for(int i=0;i<batchSize;++i) {
        cudaStreams.emplace_back(nva2x::CreateCudaStream());
        solvers.emplace_back(nva2f::CreateBlendshapeSolver(/*useGPUSolver=*/false));
        solvers.back()->SetCudaStream(cudaStreams.back()->Data());
        solvers.back()->SetJobRunner(jobRunner.get());

        if (solvers.back()->SetBlendshapeData(solverData->GetBlendshapeSolverDataView())) {
            std::cout << "SetBlendshapeData failed." << std::endl;
            return;
        }
        if (solvers.back()->Prepare()) {
            std::cout << "Prepare failed." << std::endl;
            return;
        }
    }
    auto targetPose = generateTargetPose(solvers[0]);
    if (targetPose.empty()) {
        std::cout << "Generate targetPose failed." << std::endl;
        return;
    }
    for(int i=0;i<batchSize;++i) {
        if (targetPose.size() % solvers[i]->PoseSize() != 0) {
            std::cout << "Size mismatch." << std::endl;
            return;
        }
    }


    std::ostringstream label;
    label << "identity: " << testParam.identity
          << ", geometry: " << Geometry2NpzKey[testParam.geometryType];
    state.SetLabel(label.str());

    const size_t poseSize = solvers[0]->PoseSize();
    const size_t nbFrames = targetPose.size() / poseSize;
    std::vector<UniquePtr<nva2x::IDeviceTensorFloat>> targetPoseDeviceBatch;
    std::vector<UniquePtr<nva2x::IHostTensorFloat>> solvedWeightsBatch;
    for(int i=0;i<batchSize;++i) {
        targetPoseDeviceBatch.emplace_back(nva2x::CreateDeviceTensorFloat({targetPose.data(), targetPose.size()}, cudaStreams[i]->Data()));
        solvedWeightsBatch.emplace_back(nva2x::CreateHostTensorFloat(solvers[i]->NumBlendshapePoses()));
    }

    // warm-up
    {
        for(int i=0;i<solvers.size();++i) {
            if (solvers[i]->Solve(targetPoseDeviceBatch[i]->View(0, poseSize), *(solvedWeightsBatch[i]))) {
                state.SkipWithError("Warmup failed.");
            }
        }
    }

    int idx = 0;
    for (auto _ : state) {
        std::vector<std::promise<std::error_code>> promises(batchSize);
        std::vector<std::future<std::error_code>> futures;
        for(int i=0;i<batchSize;++i) {
            futures.emplace_back(promises[i].get_future());
        }

        // Run SolveAsync in parallel for batchSize times
        for(int i=0;i<batchSize;++i) {
            if (solvers[i]->SolveAsync(targetPoseDeviceBatch[i]->View(idx * poseSize, poseSize), *(solvedWeightsBatch[i]), [](void* p, std::error_code error) {
                reinterpret_cast<std::promise<std::error_code>*>(p)->set_value(error);
            }, &promises[i])) {
                state.SkipWithError("SolveAsync failed.");
            }
        }
        idx = (idx + 1) % nbFrames;
        // Wait
        for (auto& future : futures) {
            if (future.wait_for(std::chrono::seconds(3)) != std::future_status::ready) {
                state.SkipWithError("Blendshape solve did not complete in 3 seconds");
            }
            if (future.get()) {
                state.SkipWithError("SolveAsync callback returned error.");
            }
        }
    }
    state.SetItemsProcessed(batchSize * state.iterations()); // Ensure the time reported is still for "one" SolveAsync
}
BENCHMARK_REGISTER_F(BlendshapeSolverBatchBenchmark, BM_CPUSolveAsync)->Apply([](benchmark::internal::Benchmark* b){
    b->UseRealTime(); // Configures the benchmark to use real-time (wall clock) instead of CPU time.
                  // This is necessary for the CPU solver, which operates in a multi-threaded environment,
                  // as CPU time measurement may not accurately reflect execution duration when multiple
                  // threads are involved.
    for(int batchSize : {1, 2, 5, 10}) {
        for(int j=0;j<testParams.size();++j) {
            b->Args({batchSize, j});
        }
    }
});

BENCHMARK_DEFINE_F(BlendshapeSolverBatchBenchmark, BM_GPUSolveAsync)(benchmark::State& state) {
    auto solverData = ToUniquePtr(nva2f::ReadBlendshapeSolverData(testParam.bsDataPath.data()));
    if (!solverData) {
        std::cout << "ReadBlendshapeSolverData failed." << std::endl;
        return;
    }
    std::vector<UniquePtr<nva2x::ICudaStream>> cudaStreams;
    std::vector<UniquePtr<nva2f::IBlendshapeSolver>> solvers;
    for(int i=0;i<batchSize;++i) {
        cudaStreams.emplace_back(nva2x::CreateCudaStream());
        solvers.emplace_back(nva2f::CreateBlendshapeSolver(/*useGPUSolver=*/true));
        solvers.back()->SetCudaStream(cudaStreams.back()->Data());

        if (solvers.back()->SetBlendshapeData(solverData->GetBlendshapeSolverDataView())) {
            std::cout << "SetBlendshapeData failed." << std::endl;
            return;
        }
        if (solvers.back()->Prepare()) {
            std::cout << "Prepare failed." << std::endl;
            return;
        }
    }
    auto targetPose = generateTargetPose(solvers[0]);
    if (targetPose.empty()) {
        std::cout << "Generate targetPose failed." << std::endl;
        return;
    }
    for(int i=0;i<batchSize;++i) {
        if (targetPose.size() % solvers[i]->PoseSize() != 0) {
            std::cout << "Size mismatch." << std::endl;
            return;
        }
    }


    std::ostringstream label;
    label << "identity: " << testParam.identity
          << ", geometry: " << Geometry2NpzKey[testParam.geometryType];
    state.SetLabel(label.str());

    const size_t poseSize = solvers[0]->PoseSize();
    const size_t nbFrames = targetPose.size() / poseSize;
    std::vector<UniquePtr<nva2x::IDeviceTensorFloat>> targetPoseDeviceBatch;
    std::vector<UniquePtr<nva2x::IDeviceTensorFloat>> solvedWeightsDeviceBatch;
    std::vector<UniquePtr<nva2x::IHostTensorFloat>> solvedWeightsBatch;
    for(int i=0;i<batchSize;++i) {
        targetPoseDeviceBatch.emplace_back(nva2x::CreateDeviceTensorFloat({targetPose.data(), targetPose.size()}, cudaStreams[i]->Data()));
        solvedWeightsDeviceBatch.emplace_back(nva2x::CreateDeviceTensorFloat(solvers[i]->NumBlendshapePoses()));
        solvedWeightsBatch.emplace_back(nva2x::CreateHostTensorFloat(solvers[i]->NumBlendshapePoses()));
    }

    // warm-up
    {
        for(int i=0;i<solvers.size();++i) {
            if (solvers[i]->Solve(targetPoseDeviceBatch[i]->View(0, poseSize), *(solvedWeightsBatch[i]))) {
                state.SkipWithError("Warmup failed.");
            }
        }
    }

    int idx = 0;
    for (auto _ : state) {
        // Run SolveAsync in parallel for batchSize times
        for(int i=0;i<batchSize;++i) {
            if (solvers[i]->SolveAsync(targetPoseDeviceBatch[i]->View(idx * poseSize, poseSize), *solvedWeightsDeviceBatch[i])) {
                state.SkipWithError("SolveAsync failed.");
            }
            if (nva2x::CopyDeviceToHost(*solvedWeightsBatch[i], *solvedWeightsDeviceBatch[i], cudaStreams[i]->Data())) {
                state.SkipWithError("CopyDeviceToHost failed.");
            }
        }
        idx = (idx + 1) % nbFrames;
        // Ensure the output is to a host buffer for a fair comparison with BM_CPUSolveAsync
        // Wait
        for(int i=0;i<batchSize;++i) {
            cudaStreams[i]->Synchronize();
        }
    }
    state.SetItemsProcessed(batchSize * state.iterations()); // Ensure the time reported is still for "one" SolveAsync
}
BENCHMARK_REGISTER_F(BlendshapeSolverBatchBenchmark, BM_GPUSolveAsync)->Apply([](benchmark::internal::Benchmark* b){
    b->UseRealTime(); // Configures the benchmark to use real-time (wall clock) instead of CPU time.
                  // This is necessary for the CPU solver, which operates in a multi-threaded environment,
                  // as CPU time measurement may not accurately reflect execution duration when multiple
                  // threads are involved.
    for(int batchSize : {1, 2, 5, 10, 20, 50}) {
        for(int j=0;j<testParams.size();++j) {
            b->Args({batchSize, j});
        }
    }
});
