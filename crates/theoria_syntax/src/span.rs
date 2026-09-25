//! Byte-offset source spans.
//!
//! A [`Span`] is a half-open byte range into the source text. Offsets are
//! `u32`: a 4 GB source file is not a real concern, and `u32` halves the
//! size of every CST node relative to `usize`.

/// A half-open byte range into the source.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Span {
    /// Inclusive start offset.
    pub start: u32,
    /// Exclusive end offset.
    pub end: u32,
}

impl Span {
    /// Construct a span from byte offsets.
    pub const fn new(start: u32, end: u32) -> Self {
        Span { start, end }
    }

    /// An empty span at the given offset.
    pub const fn empty(at: u32) -> Self {
        Span { start: at, end: at }
    }

    /// The union of two spans.
    ///
    /// Empty spans are absorbed idempotently: merging with an empty span
    /// returns the other span unchanged.
    ///
    /// # Panics
    ///
    /// Panics if both spans are non-empty and neither overlap nor touch
    /// (`self.end >= other.start && other.end >= self.start` must hold).
    /// Contiguous construction is a parser-internal invariant.
    pub fn merge(self, other: Span) -> Span {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }
        assert!(
            self.start <= other.end && other.start <= self.end,
            "Span::merge of disjoint spans"
        );
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Length in bytes.
    pub fn len(self) -> u32 {
        self.end.saturating_sub(self.start)
    }

    /// `true` iff the span covers no bytes.
    pub fn is_empty(self) -> bool {
        self.start >= self.end
    }

    /// The smallest span covering both inputs, without any adjacency
    /// requirement.
    ///
    /// Unlike [`merge`](Span::merge), this never panics: it is intended
    /// for parser nodes, where `from` is the first token consumed and
    /// `to` is the last, and whitespace or operators may lie between
    /// them. Empty inputs are absorbed as in `merge`.
    pub fn covering(from: Span, to: Span) -> Span {
        if from.is_empty() {
            return to;
        }
        if to.is_empty() {
            return from;
        }
        Span {
            start: from.start.min(to.start),
            end: from.end.max(to.end),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_adjacent_and_overlapping() {
        assert_eq!(Span::new(0, 2).merge(Span::new(2, 5)), Span::new(0, 5));
        assert_eq!(Span::new(0, 3).merge(Span::new(2, 5)), Span::new(0, 5));
        assert_eq!(Span::new(4, 4).merge(Span::new(4, 4)), Span::new(4, 4));
    }

    #[test]
    fn merge_identical() {
        let s = Span::new(3, 7);
        assert_eq!(s.merge(s), s);
    }

    #[test]
    fn merge_empty_is_idempotent() {
        let s = Span::new(3, 7);
        assert_eq!(s.merge(Span::empty(100)), s);
        assert_eq!(Span::empty(100).merge(s), s);
        assert_eq!(Span::empty(1).merge(Span::empty(2)), Span::empty(2));
    }

    #[test]
    #[should_panic(expected = "disjoint")]
    fn merge_disjoint_panics() {
        let _ = Span::new(0, 2).merge(Span::new(5, 7));
    }

    #[test]
    fn len_and_is_empty() {
        assert_eq!(Span::new(3, 7).len(), 4);
        assert_eq!(Span::empty(3).len(), 0);
        assert!(Span::empty(3).is_empty());
        assert!(!Span::new(3, 4).is_empty());
    }

    #[test]
    fn covering_spans_gaps() {
        assert_eq!(
            Span::covering(Span::new(0, 6), Span::new(10, 11)),
            Span::new(0, 11)
        );
        assert_eq!(
            Span::covering(Span::empty(5), Span::new(1, 2)),
            Span::new(1, 2)
        );
    }
}
