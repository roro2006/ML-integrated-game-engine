//! A minimal blocking executor.
//!
//! wgpu's adapter and device requests are async but resolve immediately on
//! native backends. Rather than adopt an async runtime for two calls, the
//! renderer parks the calling thread until the future completes.

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::thread::{self, Thread};

struct ThreadWaker(Thread);

impl Wake for ThreadWaker {
    fn wake(self: Arc<Self>) {
        self.0.unpark();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.unpark();
    }
}

pub(crate) fn block_on<F: Future>(future: F) -> F::Output {
    let waker = Waker::from(Arc::new(ThreadWaker(thread::current())));
    let mut context = Context::from_waker(&waker);
    let mut future = pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            Poll::Ready(output) => return output,
            Poll::Pending => thread::park(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_ready_futures() {
        assert_eq!(block_on(std::future::ready(7)), 7);
    }

    #[test]
    fn resolves_futures_woken_from_another_thread() {
        let (sender, receiver) = std::sync::mpsc::channel();
        let handle = thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(10));
            let _ = sender.send(3);
        });
        let value = block_on(std::future::poll_fn(|context| match receiver.try_recv() {
            Ok(value) => Poll::Ready(value),
            Err(_) => {
                let waker = context.waker().clone();
                thread::spawn(move || {
                    thread::sleep(std::time::Duration::from_millis(1));
                    waker.wake();
                });
                Poll::Pending
            }
        }));
        assert_eq!(value, 3);
        handle.join().unwrap();
    }
}
