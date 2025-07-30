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

#include <cstdint>
#include <AudioFile.h>

std::vector<float> upsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) {
    std::vector<float> output;
    float ratio = static_cast<float>(targetSampleRate) / originalSampleRate;

    for (size_t i = 0; i < input.size(); ++i) {
        output.push_back(input[i]);
        if (i < input.size() - 1) {
            float nextSample = input[i + 1];
            for (float t = 1.0f; t < ratio; t += 1.0f) {
                float interpolatedSample = input[i] + (nextSample - input[i]) * (t / ratio);
                output.push_back(interpolatedSample);
            }
        }
    }
    return output;
}

std::vector<float> downsample(const std::vector<float>& input, int targetSampleRate, int originalSampleRate) { //decimate
    std::vector<float> output;
    float ratio = static_cast<float>(originalSampleRate) / targetSampleRate;

    for (size_t i = 0; i < input.size(); i += static_cast<size_t>(ratio)) {
        output.push_back(input[i]);
    }

    return output;
}

std::vector<float> readAudio(const std::string& filename) {
    AudioFile<float> audio(filename);
    if(audio.getNumChannels() == 0 || audio.getLengthInSeconds() == 0) return {};
    const auto sr = audio.getSampleRate();
    // FIXME: Hard-coded number of samples, we should use audio_params.samplerate from the network info.
    if(sr == 16000) return audio.samples[0];
    const auto original = audio.samples[0];

    if(sr < 16000) {
        std::cerr << "Unsupported sample rate " << sr << std::endl;
        return {};
    }

        //really bad resampling, let's use matx poly_resample, which is the sampe implementation of scipy poly resample
    const int multiple = sr/16000;
    if(multiple * 16000 == sr) // multiple of 16khz khz
    {
        return downsample(original, 16000, sr);
    }
    if(audio.getSampleRate() == 24000) // 44.1 khz
    {
        const int lcm  = 48000;
        return downsample(upsample(original,  lcm, sr), 16000, lcm);
    }
    if(sr == 44100 || sr == 88200) // 44.1 khz 88.2khz
    {
        const int lcm  = 7056000;
        return downsample(upsample(original,  lcm, sr), 16000, lcm);
    }

    std::cerr << "Unsupported sample rate " << sr << std::endl;
    return {}; //not supported
}

std::vector<float> loadAudio() {
    // OPTME: allow for switching audio track
    return readAudio(TEST_DATA_DIR "sample-data/audio_4sec_16k_s16le.wav");
}

void AddDefaultEmotion(benchmark::State& state, nva2f::IGeometryExecutorBundle& bundle) {
    const auto nbTracks = bundle.GetExecutor().GetNbTracks();
    std::vector<float> emptyEmotion;
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        auto& emotionAccumulator = bundle.GetEmotionAccumulator(trackIndex);
        emptyEmotion.resize(emotionAccumulator.GetEmotionSize(), 0.0f);
        CHECK_AND_SKIP(!emotionAccumulator.Accumulate(
            0, nva2x::HostTensorFloatConstView{emptyEmotion.data(), emptyEmotion.size()}, bundle.GetCudaStream().Data()
            ));
        CHECK_AND_SKIP(!emotionAccumulator.Close());
    }
}

void AddDefaultEmotion(benchmark::State& state, nva2f::IBlendshapeExecutorBundle& bundle) {
    const auto nbTracks = bundle.GetExecutor().GetNbTracks();
    std::vector<float> emptyEmotion;
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        auto& emotionAccumulator = bundle.GetEmotionAccumulator(trackIndex);
        emptyEmotion.resize(emotionAccumulator.GetEmotionSize(), 0.0f);
        CHECK_AND_SKIP(!emotionAccumulator.Accumulate(
            0, nva2x::HostTensorFloatConstView{emptyEmotion.data(), emptyEmotion.size()}, bundle.GetCudaStream().Data()
            ));
        CHECK_AND_SKIP(!emotionAccumulator.Close());
    }
}

TimePoint startTimer() {
    return Clock::now();
}

double getElapsedMilliseconds(const TimePoint& startTime) {
    return std::chrono::duration<double, std::milli>(Clock::now() - startTime).count();
}

// GeometryExecutorResultsCollector implementations
void GeometryExecutorResultsCollector::Init(nva2f::IGeometryExecutorBundle* bundle, benchmark::State& state) {
    _bundle = bundle;
    CHECK_AND_SKIP(!_bundle->GetExecutor().SetResultsCallback(callbackForGeometryExecutor, &_callbackData));
    ResetCounters();
}

bool GeometryExecutorResultsCollector::callbackForGeometryExecutor(void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& data = *static_cast<GeometryExecutorCallbackData*>(userdata);
    data.frameIndices[results.trackIndex] += 1;
    return true;
}

void GeometryExecutorResultsCollector::ResetCounters() {
    _callbackData.frameIndices.clear();
    _callbackData.frameIndices.resize(_bundle->GetExecutor().GetNbTracks(), 0);
}

std::size_t GeometryExecutorResultsCollector::GetTotalFrames() const {
    return std::accumulate(_callbackData.frameIndices.begin(), _callbackData.frameIndices.end(), 0);
}

bool GeometryExecutorResultsCollector::HasFrameGenerated(std::size_t trackIndex) const {
    return _callbackData.frameIndices[trackIndex] > 0;
}

bool GeometryExecutorResultsCollector::Wait() {
    if (_bundle->GetCudaStream().Synchronize()) {
        return false;
    }
    return true;
}

// BlendshapeSolveExecutorResultsCollector implementations
void BlendshapeSolveExecutorResultsCollector::Init(nva2f::IBlendshapeExecutorBundle* bundle, benchmark::State& state) {
    _bundle = bundle;
    _callbackData.state = &state;
    _callbackData.weightViews.resize(_bundle->GetExecutor().GetNbTracks(), {});
    auto& executor = _bundle->GetExecutor();
    if (executor.GetResultType() == nva2f::IBlendshapeExecutor::ResultsType::HOST) {
        auto callback = [](void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode) -> void {
            callbackForHostBlendshapeSolveExecutor(userdata, results, errorCode);
        };
        CHECK_AND_SKIP(!executor.SetResultsCallback(callback, &_callbackData));
    } else if (executor.GetResultType() == nva2f::IBlendshapeExecutor::ResultsType::DEVICE) {
        _weightHostPinnedBatch.clear();
        for (std::size_t trackIndex = 0; trackIndex < executor.GetNbTracks(); ++trackIndex) {
            _weightHostPinnedBatch.emplace_back(nva2x::CreateHostPinnedTensorFloat(executor.GetWeightCount()));
            _callbackData.weightViews[trackIndex] = *(_weightHostPinnedBatch.back());
        }
        CHECK_AND_SKIP(!executor.SetResultsCallback(callbackForDeviceBlendshapeSolveExecutor, &_callbackData));
    } else {
        state.SkipWithError("Unknown results type.");
    }
    ResetCounters();
}

void BlendshapeSolveExecutorResultsCollector::callbackForHostBlendshapeSolveExecutor(void* userdata, const nva2f::IBlendshapeExecutor::HostResults& results, std::error_code errorCode) {
    auto& data = *static_cast<BlendshapeSolveExecutorCallbackData*>(userdata);
    data.frameIndices[results.trackIndex] += 1;
}

bool BlendshapeSolveExecutorResultsCollector::callbackForDeviceBlendshapeSolveExecutor(void* userdata, const nva2f::IBlendshapeExecutor::DeviceResults& results) {
    auto& data = *static_cast<BlendshapeSolveExecutorCallbackData*>(userdata);
    auto& state = *data.state;
    // copy to pinned host buffer for a fair comparison
    if (data.weightViews[results.trackIndex].Size() > 0 && results.weights.Size() > 0) {
        CHECK_AND_SKIP(!nva2x::CopyDeviceToHost(data.weightViews[results.trackIndex], results.weights, results.cudaStream));
        data.frameIndices[results.trackIndex] += 1;
    }
    return true;
}

void BlendshapeSolveExecutorResultsCollector::ResetCounters() {
    _callbackData.frameIndices.clear();
    _callbackData.frameIndices.resize(_bundle->GetExecutor().GetNbTracks(), 0);
}

std::size_t BlendshapeSolveExecutorResultsCollector::GetTotalFrames() const {
    return std::accumulate(_callbackData.frameIndices.begin(), _callbackData.frameIndices.end(), 0);
}

bool BlendshapeSolveExecutorResultsCollector::HasFrameGenerated(std::size_t trackIndex) const {
    return _callbackData.frameIndices[trackIndex] > 0;
}

bool BlendshapeSolveExecutorResultsCollector::Wait() {
    std::size_t nbTracks =  _bundle->GetExecutor().GetNbTracks();
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        if (_bundle->GetExecutor().Wait(trackIndex)) {
            return false;
        }
    }
    return true;
}

// EmotionExecutorResultsCollector implementations
template <typename ExecutorBundleType>
void EmotionExecutorResultsCollector::Init(nva2e::IEmotionExecutor* executor, ExecutorBundleType* executorBundle, benchmark::State& state) {
    _executor = executor;
    std::vector<nva2x::IEmotionAccumulator*> localEmotionAccumulators(executorBundle->GetExecutor().GetNbTracks());
    for (std::size_t trackIndex = 0; trackIndex < executorBundle->GetExecutor().GetNbTracks(); ++trackIndex) {
        localEmotionAccumulators[trackIndex] = &executorBundle->GetEmotionAccumulator(trackIndex);
    }
    _callbackData.emotionAccumulators = std::move(localEmotionAccumulators);
    _callbackData.state = &state;
    ResetCounters();
    CHECK_AND_SKIP(!_executor->SetResultsCallback(callbackForEmotionExecutor, &_callbackData));
}

bool EmotionExecutorResultsCollector::callbackForEmotionExecutor(void* userdata, const nva2e::IEmotionExecutor::Results& results) {
    auto& data = *static_cast<EmotionExecutorCallbackData*>(userdata);
    data.frameIndices[results.trackIndex] += 1;
    auto& state = *data.state;
    CHECK_AND_SKIP(!data.emotionAccumulators[results.trackIndex]->Accumulate(results.timeStampCurrentFrame, results.emotions, results.cudaStream));
    return true;
}

void EmotionExecutorResultsCollector::ResetCounters() {
    _callbackData.frameIndices.clear();
    _callbackData.frameIndices.resize(_executor->GetNbTracks(), 0);
}

std::size_t EmotionExecutorResultsCollector::GetTotalFrames() const {
    return std::accumulate(_callbackData.frameIndices.begin(), _callbackData.frameIndices.end(), 0);
}

bool EmotionExecutorResultsCollector::HasFrameGenerated(std::size_t trackIndex) const {
    return _callbackData.frameIndices[trackIndex] > 0;
}

// RunExecutorOffline implementation
template<typename A2FExecutorBundleType>
void RunExecutorOffline(
    benchmark::State& state,
    bool precomputeA2E,
    UniquePtr<A2FExecutorBundleType>& a2fExecutorBundle,
    UniquePtr<nva2e::IEmotionExecutor>& emotionExecutor
) {
    const auto nbTracks = a2fExecutorBundle->GetExecutor().GetNbTracks();
    assert(emotionExecutor->GetNbTracks() == nbTracks);

    using A2FResultsCollectorType = std::conditional_t<
        std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>,
        GeometryExecutorResultsCollector,
        BlendshapeSolveExecutorResultsCollector
    >;
    A2FResultsCollectorType a2fExecutorResultsCollector;
    a2fExecutorResultsCollector.Init(a2fExecutorBundle.get(), state);

    EmotionExecutorResultsCollector emotionExecutorResultsCollector;
    emotionExecutorResultsCollector.Init<A2FExecutorBundleType>(emotionExecutor.get(), a2fExecutorBundle.get(), state);

    // Then, load all the audio and accumulate it.
    const auto audioBuffer = loadAudio();
    CHECK_AND_SKIP(!audioBuffer.empty());
    for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
        // We put same amount of audio in each track to test the executor scalability
        CHECK_AND_SKIP(
            !a2fExecutorBundle->GetAudioAccumulator(trackIndex).Accumulate(
            nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size()}, a2fExecutorBundle->GetCudaStream().Data()
            )
            );
        CHECK_AND_SKIP(!a2fExecutorBundle->GetAudioAccumulator(trackIndex).Close());
    }

    // warm-up
    // Run until at least one frame is available, because execution for diffusion
    // can return 0 frames for the first execution in the padding before the audio.
    while (!a2fExecutorResultsCollector.HasFrameGenerated(0)) {
        while (!(nva2x::GetNbReadyTracks(a2fExecutorBundle->GetExecutor()) > 0)) {
            CHECK_AND_SKIP(!emotionExecutor->Execute(nullptr));
        }
        CHECK_AND_SKIP(!a2fExecutorBundle->GetExecutor().Execute(nullptr));
        CHECK_AND_SKIP(!a2fExecutorBundle->GetCudaStream().Synchronize());
    }
    a2fExecutorResultsCollector.ResetCounters();
    emotionExecutorResultsCollector.ResetCounters();

    for (auto _ : state) {
        state.PauseTiming();
        CHECK_AND_SKIP(!a2fExecutorBundle->GetCudaStream().Synchronize());
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!a2fExecutorBundle->GetExecutor().Reset(trackIndex));
            CHECK_AND_SKIP(!a2fExecutorBundle->GetEmotionAccumulator(trackIndex).Reset());
            CHECK_AND_SKIP(!emotionExecutor->Reset(trackIndex));
        }
        CHECK_AND_SKIP(!a2fExecutorBundle->GetCudaStream().Synchronize());
        // Process all emotion
        {
            if (!precomputeA2E) {
                // If not precomputing A2E, we can resume timing right away.
                state.ResumeTiming();
            }
            auto startTimeA2E = startTimer();
            while (nva2x::GetNbReadyTracks(*emotionExecutor) > 0) {
                CHECK_AND_SKIP(!emotionExecutor->Execute(nullptr));
            }
            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                CHECK_AND_SKIP(!a2fExecutorBundle->GetEmotionAccumulator(trackIndex).Close());
            }
            state.counters["A2ETotalTime(ms)"] = getElapsedMilliseconds(startTimeA2E);
            std::size_t totalFrames = emotionExecutorResultsCollector.GetTotalFrames();
            state.counters["A2EAvgMultiTrackProcessingTime(ms)"] = state.counters["A2ETotalTime(ms)"] / totalFrames * nbTracks;
            state.counters["A2EAvgPerTrackProcessingTime(ms)"] = state.counters["A2ETotalTime(ms)"] / totalFrames;
            if (precomputeA2E) {
                // If precomputing A2E, we need to wait for the A2E to be done before resuming timing.
                state.ResumeTiming();
            }
        }
        // Process all geometry
        auto startTimeA2F = startTimer();
        while (nva2x::GetNbReadyTracks(a2fExecutorBundle->GetExecutor()) > 0) {
            CHECK_AND_SKIP(!a2fExecutorBundle->GetExecutor().Execute(nullptr));
        }
        state.counters["A2FExecuteTime(ms)"] = getElapsedMilliseconds(startTimeA2F);
        CHECK_AND_SKIP(a2fExecutorResultsCollector.Wait());
        state.counters["A2FTotalTime(ms)"] = getElapsedMilliseconds(startTimeA2F);
    }

    std::size_t totalFrames = a2fExecutorResultsCollector.GetTotalFrames();
    state.SetItemsProcessed(totalFrames);
    state.counters["A2FAvgMultiTrackProcessingTime(ms)"] = state.counters["A2FTotalTime(ms)"] / totalFrames * nbTracks;
    state.counters["A2FAvgPerTrackProcessingTime(ms)"] = state.counters["A2FTotalTime(ms)"] / totalFrames;
    state.counters["TotalTime(ms)"] = state.counters["A2ETotalTime(ms)"] + state.counters["A2FTotalTime(ms)"];
    state.counters["nbTracks"] = static_cast<double>(nbTracks); // state.counters only accepts double
}

// RunExecutorStreaming implementation
template<typename A2FExecutorBundleType>
void RunExecutorStreaming(
    benchmark::State& state,
    std::size_t audioChunkSize,
    UniquePtr<A2FExecutorBundleType>& bundle,
    UniquePtr<nva2e::IEmotionExecutor>& emotionExecutor
) {
    assert(audioChunkSize > 0);
    auto& executor = bundle->GetExecutor();
    const auto nbTracks = executor.GetNbTracks();
    assert(emotionExecutor->GetNbTracks() == nbTracks);

    using A2FResultsCollectorType = std::conditional_t<
        std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>,
        GeometryExecutorResultsCollector,
        BlendshapeSolveExecutorResultsCollector
    >;
    A2FResultsCollectorType a2fExecutorResultsCollector;
    a2fExecutorResultsCollector.Init(bundle.get(), state);

    EmotionExecutorResultsCollector emotionExecutorResultsCollector;
    emotionExecutorResultsCollector.Init<A2FExecutorBundleType>(emotionExecutor.get(), bundle.get(), state);

    // Load all the audio, but don't accumulate it yet.
    const auto audioBuffer = loadAudio();
    CHECK_AND_SKIP(!audioBuffer.empty());

    // Processing tries to do as little emotion processing to unblock geometry
    // processing which requires emotion results, so that emotion processing can
    // run as quickly as possible.
    auto processAvailableData = [&]() {
        // Note that in multi-track mode, we might want to only run when at least
        // a certain number of tracks are ready to maximize parallelism.

        while (true) {
            // Process available blendshape
            const auto nbBlendshapeTracks = nva2x::GetNbReadyTracks(bundle->GetExecutor());
            if (nbBlendshapeTracks > 0) {
                CHECK_AND_SKIP(!bundle->GetExecutor().Execute(nullptr));
                continue;
            }

            // No blendshape can be processed, process emotion which might enable further blendshape processing.
            const auto nbEmotionTracks = nva2x::GetNbReadyTracks(*emotionExecutor);
            if (nbEmotionTracks > 0) {
                CHECK_AND_SKIP(!emotionExecutor->Execute(nullptr));
                continue;
            }

            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                if (bundle->GetAudioAccumulator(trackIndex).IsClosed()) {
                    if (!bundle->GetEmotionAccumulator(trackIndex).IsClosed()) {
                        // We are done accumulating audio.  If there is no more emotion to process,
                        // it means we are done accumulating emotions.
                        CHECK_AND_SKIP(!bundle->GetEmotionAccumulator(trackIndex).Close());
                    }
                }
            }

            // No more emotion or blendshape to process, exit.
            break;
        }
    };

    // warm-up
    // Run until at least one frame is available, because execution for diffusion
    // can return 0 frames for the first execution in the padding before the audio.
    for (std::size_t i = 0; i < audioBuffer.size() && (!a2fExecutorResultsCollector.HasFrameGenerated(0)); i += audioChunkSize) {
        const auto chunkData = audioBuffer.data() + i;
        const auto chunkSize = std::min(audioChunkSize, audioBuffer.size() - i);
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Accumulate(
                nva2x::HostTensorFloatConstView{chunkData, chunkSize}, bundle->GetCudaStream().Data()
                )
            );
        }
        // Process available data.
        processAvailableData();
    }
    a2fExecutorResultsCollector.ResetCounters();
    emotionExecutorResultsCollector.ResetCounters();

    for (auto _ : state) {
        state.PauseTiming();
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!executor.Reset(trackIndex));
            CHECK_AND_SKIP(!bundle->GetEmotionAccumulator(trackIndex).Reset());
            CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Reset());
            CHECK_AND_SKIP(!emotionExecutor->Reset(trackIndex));
        }
        if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IBlendshapeExecutorBundle>) {
            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                CHECK_AND_SKIP(!executor.Wait(trackIndex));
            }
        } else if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>) {
            CHECK_AND_SKIP(!bundle->GetCudaStream().Synchronize());
        }
        state.ResumeTiming();
        auto startTime = startTimer();
        for (std::size_t i = 0; i < audioBuffer.size(); i += audioChunkSize) {
            const auto chunkData = audioBuffer.data() + i;
            const auto chunkSize = std::min(audioChunkSize, audioBuffer.size() - i);
            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Accumulate(
                    nva2x::HostTensorFloatConstView{chunkData, chunkSize}, bundle->GetCudaStream().Data()
                    )
                );
            }
            // Process available data.
            processAvailableData();
        }
        for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
            CHECK_AND_SKIP(!bundle->GetAudioAccumulator(trackIndex).Close());
        }
        // After closing the audio, we might be able to do more processing.
        processAvailableData();
        if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IBlendshapeExecutorBundle>) {
            for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
                CHECK_AND_SKIP(!executor.Wait(trackIndex));
            }
        } else if constexpr (std::is_same_v<A2FExecutorBundleType, nva2f::IGeometryExecutorBundle>) {
            CHECK_AND_SKIP(!bundle->GetCudaStream().Synchronize());
        }
        state.counters["TotalTime(ms)"] = getElapsedMilliseconds(startTime);
    }

    std::size_t totalFrames = a2fExecutorResultsCollector.GetTotalFrames();
    state.SetItemsProcessed(totalFrames);
    state.counters["AvgMultiTrackProcessingTime(ms)"] = state.counters["TotalTime(ms)"] / totalFrames * nbTracks;
    state.counters["AvgPerTrackProcessingTime(ms)"] = state.counters["TotalTime(ms)"] / totalFrames;
    state.counters["nbTracks"] = static_cast<double>(nbTracks); // state.counters only accepts double
}

// Explicit template instantiations
template void RunExecutorOffline<nva2f::IGeometryExecutorBundle>(
    benchmark::State& state,
    bool precomputeA2E,
    UniquePtr<nva2f::IGeometryExecutorBundle>& a2fExecutorBundle,
    UniquePtr<nva2e::IEmotionExecutor>& emotionExecutor
);

template void RunExecutorOffline<nva2f::IBlendshapeExecutorBundle>(
    benchmark::State& state,
    bool precomputeA2E,
    UniquePtr<nva2f::IBlendshapeExecutorBundle>& a2fExecutorBundle,
    UniquePtr<nva2e::IEmotionExecutor>& emotionExecutor
);

template void RunExecutorStreaming<nva2f::IGeometryExecutorBundle>(
    benchmark::State& state,
    std::size_t audioChunkSize,
    UniquePtr<nva2f::IGeometryExecutorBundle>& bundle,
    UniquePtr<nva2e::IEmotionExecutor>& emotionExecutor
);

template void RunExecutorStreaming<nva2f::IBlendshapeExecutorBundle>(
    benchmark::State& state,
    std::size_t audioChunkSize,
    UniquePtr<nva2f::IBlendshapeExecutorBundle>& bundle,
    UniquePtr<nva2e::IEmotionExecutor>& emotionExecutor
);

template<typename BundleType>
UniquePtr<nva2e::IEmotionExecutor> CreateEmotionExecutor(
    cudaStream_t cudaStream, UniquePtr<BundleType>& bundle, std::size_t inferencesToSkip
) {
    auto modelInfo = ToUniquePtr(nva2e::ReadClassifierModelInfo(
        TEST_DATA_DIR "_data/generated/audio2emotion-sdk/samples/model/model.json"
    ));
    if (!modelInfo) {
        return nullptr;
    }

    // Get all audio accumulators pointers.
    const std::vector<const nva2x::IAudioAccumulator*> audioAccumulators = [&]() {
        std::vector<const nva2x::IAudioAccumulator*> audioAccumulators;
        for(std::size_t i=0;i<bundle->GetExecutor().GetNbTracks();++i) {
            audioAccumulators.push_back(&bundle->GetAudioAccumulator(i));
        }
        return audioAccumulators;
    }();
    assert(audioAccumulators.size() == bundle->GetExecutor().GetNbTracks());

    nva2e::EmotionExecutorCreationParameters params;
    params.cudaStream = cudaStream;
    params.nbTracks = bundle->GetExecutor().GetNbTracks();
    params.sharedAudioAccumulators = audioAccumulators.data();

    // Notice that it produces frames at a different frame rate than the geometry executor.
    auto classifierParams = modelInfo->GetExecutorCreationParameters(
        60000, 30, 1, inferencesToSkip
    );

    auto executor = ToUniquePtr(nva2e::CreateClassifierEmotionExecutor(params, classifierParams));
    if (!executor) {
        return nullptr;
    }

    return executor;
}

// Explicit template instantiations for the types that will be used
template UniquePtr<nva2e::IEmotionExecutor> CreateEmotionExecutor<nva2f::IGeometryExecutorBundle>(
    cudaStream_t cudaStream, UniquePtr<nva2f::IGeometryExecutorBundle>& bundle, std::size_t inferencesToSkip);
template UniquePtr<nva2e::IEmotionExecutor> CreateEmotionExecutor<nva2f::IBlendshapeExecutorBundle>(
    cudaStream_t cudaStream, UniquePtr<nva2f::IBlendshapeExecutorBundle>& bundle, std::size_t inferencesToSkip);
