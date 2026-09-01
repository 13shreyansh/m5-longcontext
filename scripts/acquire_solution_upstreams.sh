#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
UPSTREAM_ROOT="$ROOT/artifacts/upstreams"

acquire() {
    name=$1
    url=$2
    commit=$3
    destination="$UPSTREAM_ROOT/$name"

    if [ -e "$destination" ]; then
        if [ ! -d "$destination/.git" ]; then
            echo "refusing non-Git upstream path: $destination" >&2
            return 1
        fi
        actual_url=$(git -C "$destination" remote get-url origin)
        case "$actual_url" in
            "$url"|"$url.git") ;;
            *)
                echo "unexpected origin for $name: $actual_url" >&2
                return 1
                ;;
        esac
        if [ -n "$(git -C "$destination" status --porcelain)" ]; then
            echo "refusing modified upstream checkout: $destination" >&2
            return 1
        fi
        actual_commit=$(git -C "$destination" rev-parse HEAD)
        if [ "$actual_commit" != "$commit" ]; then
            echo "$name commit=$actual_commit, expected=$commit" >&2
            return 1
        fi
        echo "$name: already present at $commit"
        return 0
    fi

    temporary=$(mktemp -d "$UPSTREAM_ROOT/.${name}.XXXXXX")
    trap 'rm -rf -- "$temporary"' EXIT HUP INT TERM
    git -C "$temporary" init -q
    git -C "$temporary" remote add origin "$url"
    git -C "$temporary" fetch -q --depth 1 origin "$commit"
    git -C "$temporary" checkout -q --detach FETCH_HEAD
    actual_commit=$(git -C "$temporary" rev-parse HEAD)
    if [ "$actual_commit" != "$commit" ]; then
        echo "$name fetched commit=$actual_commit, expected=$commit" >&2
        return 1
    fi
    git -C "$temporary" fsck --no-progress --connectivity-only >/dev/null
    mv -- "$temporary" "$destination"
    trap - EXIT HUP INT TERM
    echo "$name: acquired $commit from $url"
}

mkdir -p "$UPSTREAM_ROOT"
acquire \
    triton-msl \
    https://github.com/bledden/triton-msl \
    182c1820fd24a836d565e1da842f28414de64084
acquire \
    mlx \
    https://github.com/ml-explore/mlx \
    3f0bd54ff0c0af5b88530191d5df31010ce54fcd
