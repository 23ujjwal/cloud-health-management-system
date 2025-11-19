#include <stdio.h>
#include <stdlib.h>

#define MAX 100

typedef struct {
    int data;
    int priority;
} Node;

typedef struct {
    Node nodes[MAX];
    int size;
} PriorityQueue;

void initQueue(PriorityQueue *pq) {
    pq->size = 0;
}

int isEmpty(PriorityQueue *pq) {
    return pq->size == 0;
}

void enqueue(PriorityQueue *pq, int data, int priority) {
    if (pq->size >= MAX) {
        printf("Queue is full!\n");
        return;
    }
    int i = pq->size - 1;
    while (i >= 0 && pq->nodes[i].priority < priority) {
        pq->nodes[i+1] = pq->nodes[i];
        i--;
    }
    pq->nodes[i+1].data = data;
    pq->nodes[i+1].priority = priority;
    pq->size++;
    printf("Inserted %d with priority %d\n", data, priority);
}

Node dequeue(PriorityQueue *pq) {
    Node n = {0, 0};
    if (isEmpty(pq)) {
        printf("Queue is empty!\n");
        return n;
    }
    n = pq->nodes[0];
    for (int i = 0; i < pq->size - 1; i++) {
        pq->nodes[i] = pq->nodes[i+1];
    }
    pq->size--;
    return n;
}

Node peek(PriorityQueue *pq) {
    if (isEmpty(pq)) {
        Node n = {0, 0};
        return n;
    }
    return pq->nodes[0];
}

void display(PriorityQueue *pq) {
    if (isEmpty(pq)) {
        printf("Queue is empty!\n");
        return;
    }
    printf("Queue elements (data:priority):\n");
    for (int i = 0; i < pq->size; i++) {
        printf("%d:%d ", pq->nodes[i].data, pq->nodes[i].priority);
    }
    printf("\n");
}

int main() {
    PriorityQueue pq;
    initQueue(&pq);

    enqueue(&pq, 10, 2);
    enqueue(&pq, 20, 5);
    enqueue(&pq, 30, 1);
    enqueue(&pq, 40, 4);

    display(&pq);

    Node n = dequeue(&pq);
    printf("Dequeued: %d with priority %d\n", n.data, n.priority);

    display(&pq);

    Node top = peek(&pq);
    printf("Top element: %d with priority %d\n", top.data, top.priority);

    return 0;
}
