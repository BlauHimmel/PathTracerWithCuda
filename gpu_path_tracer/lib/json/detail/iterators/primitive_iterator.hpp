#ifndef NLOHMANN_JSON_DETAIL_ITERATORS_PRIMITIVE_ITERATOR_HPP
#define NLOHMANN_JSON_DETAIL_ITERATORS_PRIMITIVE_ITERATOR_HPP

#include <ciso646> // not
#include <cstddef> // ptrdiff_t
#include <limits>  // numeric_limits
#include <ostream> // ostream

namespace nlohmann
{
namespace detail
{
/*
@brief an iterator for primitive JSON types

This class models an iterator for primitive JSON types (boolean, number,
string). It's only purpose is to allow the iterator/const_iterator classes
to "iterate" over primitive values. Internally, the iterator is modeled by
a `difference_type` variable. Value begin_value (`0`) models the begin,
end_value (`1`) models past the end.
*/
class primitive_iterator_t
{
  public:
    using difference_type = std::ptrdiff_t;

    constexpr difference_type get_value() const noexcept
    {
        return m_it;
    }

    /// set iterator to a defined beginning
    void set_begin() noexcept
    {
        m_it = begin_value;
    }

    /// set iterator to a defined past the end
    void set_end() noexcept
    {
        m_it = end_value;
    }

    /// return whether the iterator can be dereferenced
    constexpr bool is_begin() const noexcept
    {
        return m_it == begin_value;
    }

    /// return whether the iterator is at end
    constexpr bool is_end() const noexcept
    {
        return m_it == end_value;
    }

    friend constexpr bool operator==(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it == rhs.m_it;
    }

    friend constexpr bool operator<(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it < rhs.m_it;
    }

    primitive_iterator_t operator+(difference_type i)
    {
        auto result = *this;
        result += i;
        return result;
    }

    friend constexpr difference_type operator-(primitive_iterator_t lhs, primitive_iterator_t rhs) noexcept
    {
        return lhs.m_it - rhs.m_it;
    }

    friend std::ostream& operator<<(std::ostream& os, primitive_iterator_t it)
    {
        return os << it.m_it;
    }

    primitive_iterator_t& operator++()
    {
        ++m_it;
        return *this;
    }

    primitive_iterator_t const operator++(int)
    {
        auto result = *this;
        m_it++;
        return result;
    }

    primitive_iterator_t& operator--()
    {
        --m_it;
        return *this;
    }

    primitive_iterator_t const operator--(int)
    {
        auto result = *this;
        m_it--;
        return result;
    }

    primitive_iterator_t& operator+=(difference_type n)
    {
        m_it += n;
        return *this;
    }

    primitive_iterator_t& operator-=(difference_type n)
    {
        m_it -= n;
        return *this;
    }

  private:
    static constexpr difference_type begin_value = 0;
    static constexpr difference_type end_value = begin_value + 1;

    /// iterator as signed integer type
    difference_type m_it = (std::numeric_limits<std::ptrdiff_t>::min)();
};
}
}

#endif
