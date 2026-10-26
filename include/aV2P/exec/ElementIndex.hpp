#pragma once

namespace aV2P
{

    /* ElementIndex
     *
     * An aggregate that containes the `.global` and `.local` indices of an element along a given dimension.
     */

    template<typename TIdx>
    struct ElementIndex
    {
        TIdx global; // Index of the element along a given dimension, relative to the whole problem space.
        TIdx local; // Index of the element along a given dimension, relative to the current group.
    };

} // namespace aV2P
