class Heap {
  constructor(data) {
    this.heap_arr = [null, data]
  }

  moveUp(inserted_idx) {
    if (inserted_idx <= 1) {
      return false
    } else {
      const parent_idx = parseInt(inserted_idx / 2)
      if (this.heap_arr[parent_idx][0] > this.heap_arr[inserted_idx][0]) {
        return true
      } else {
        return false
      }
    }
  }

  insert(data) {
    if (this.heap_arr.length == 0) {
      this.heap_arr = [null, data]
      return
    }

    this.heap_arr.push(data)
    let inserted_idx = this.heap_arr.length - 1
    let parent_idx
    while (this.moveUp(inserted_idx)) {
      parent_idx = parseInt(inserted_idx / 2)
      const tmp = this.heap_arr[parent_idx]
      this.heap_arr[parent_idx] = this.heap_arr[inserted_idx]
      this.heap_arr[inserted_idx] = tmp
      inserted_idx = parent_idx
    }
  }

  moveDown(popped_idx) {
    const left_idx = popped_idx * 2
    const right_idx = popped_idx * 2 + 1

    if (left_idx >= this.heap_arr.length) {
      return false
    } else if (right_idx >= this.heap_arr.length) {
      if (this.heap_arr[popped_idx][0] > this.heap_arr[left_idx][0]) {
        return true
      } else {
        return false
      }
    } else {
      if (this.heap_arr[popped_idx][0] > Math.min(this.heap_arr[left_idx][0], this.heap_arr[right_idx][0])) {
        return true
      } else {
        return false
      }
    }
  }

  pop() {
    if (this.heap_arr <= 1) {
      return
    }

    const returned_data = this.heap_arr[1]
    this.heap_arr[1] = this.heap_arr[this.heap_arr.length - 1]
    this.heap_arr.pop()

    let popped_idx = 1
    while (this.moveDown(popped_idx)) {
      const left_idx = popped_idx * 2
      const right_idx = popped_idx * 2 + 1

      if (right_idx >= this.heap_arr.length) {
        const tmp = this.heap_arr[left_idx]
        this.heap_arr[left_idx] = this.heap_arr[popped_idx]
        this.heap_arr[popped_idx] = tmp
        popped_idx = left_idx
      } else {
        const idx = this.heap_arr[left_idx][0] < this.heap_arr[right_idx][0] ? left_idx : right_idx
        const tmp = this.heap_arr[idx]
        this.heap_arr[idx] = this.heap_arr[popped_idx]
        this.heap_arr[popped_idx] = tmp
        popped_idx = idx
      }
    }
    return returned_data
  }
}

const input = `1 2 2
1 3 5
1 4 1
2 3 3
2 4 2
3 2 3
3 6 5
4 3 3
4 5 1
5 3 1
5 6 2`.split('\n')

const [v, e] = [6, 11]
const start = 1
const INF = 1e9

let distance = Array.from({
  length: v + 1
}, () => INF)

let graph = Array.from({
  length: v + 1
}, () => [])

for (let i = 0; i < e; i++) {
  const [a, b, c] = input[i].trim().split(' ').map(number => parseInt(number))
  graph[a].push([b, c])
}

dijkstra(start)

console.log(distance)

function dijkstra(start) {
  const heapq = new Heap([0, start])
  distance[start] = 0

  while (heapq.heap_arr.length != 1) {
    const [dist, now] = heapq.pop()
    
    if (dist > distance[now]) {
      continue
    }

    for ([next, d] of graph[now]) {
      if (distance[next] > distance[now] + d) {
        distance[next] = distance[now] + d
        heapq.insert([distance[next], next])
      }
    }
  }
}