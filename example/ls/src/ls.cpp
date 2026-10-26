// Copyright 2023 Bernhard Manfred Gruber
// SPDX-License-Identifier: ISC

#include "aV2P/test/acc/TestAccs.hpp"

#include <aV2P/alpaka.hpp>

#include <string_view>

struct PerAcc
{
    bool verbose;

    template<typename TAcc>
    void operator()() const
    {
        auto const platform = aV2P::Platform<TAcc>{};
        std::cout << aV2P::getAccName<TAcc>() << '\n';
        for(auto const& dev : aV2P::getDevs(platform))
        {
            std::cout << '\t' << aV2P::getName(dev) << '\n';
            if(verbose)
            {
                auto const props = aV2P::getAccDevProps<TAcc>(dev);
                auto const globalMem = aV2P::getMemBytes(dev);

                std::cout << "\t\tMultiProcessorCount  " << props.m_multiProcessorCount << '\n';
                std::cout << "\t\tGlobalMemSizeBytes   " << globalMem << '\n';
                std::cout << "\t\tSharedMemSizeBytes   " << props.m_sharedMemSizeBytes << '\n';

                std::cout << "\t\tGridBlockExtentMax   " << props.m_gridBlockExtentMax << '\n';
                std::cout << "\t\tBlockThreadExtentMax " << props.m_blockThreadExtentMax << '\n';
                std::cout << "\t\tThreadElemExtentMax  " << props.m_threadElemExtentMax << '\n';

                std::cout << "\t\tGridBlockCountMax    " << props.m_gridBlockCountMax << '\n';
                std::cout << "\t\tBlockThreadCountMax  " << props.m_blockThreadCountMax << '\n';
                std::cout << "\t\tThreadElemCountMax   " << props.m_threadElemCountMax << '\n';
            }
        }
    }
};

int main(int argc, char const* argv[])
{
    auto const verbose = argc >= 2 && std::string_view{argv[1]} == "-v";

    using Idx = int;
    using Dim = aV2P::DimInt<1>;
    aV2P::meta::forEachType<aV2P::test::EnabledAccs<Dim, Idx>>(PerAcc{verbose});
}
