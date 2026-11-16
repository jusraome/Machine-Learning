"""Shared train/selection/calibration/evaluation boundaries for normal-only AE.

Labels remain the existing next-character y classes. No labels are supplied to
the decoder, no original split is changed, and test access requires opt-in.
Speaker separation prevents shared-speaker windows across validation roles; it
does not prove that identical text never occurs under different speaker IDs.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
import time

from .anomaly_evaluation import fit_normal_class


ROLE_NAMES = ("model_selection", "calibration", "evaluation")
MAX_SELECTED_ROWS = 65536


def _deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeoutError("Normal-only protocol wall-time budget exhausted")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def _positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return value


def _normal_id(pipeline, normal_class):
    if type(normal_class) is not int or not 0 <= normal_class < len(pipeline.vocab["itos"]):
        raise ValueError("normal_class must be an existing vocabulary ID")
    return normal_class


def _speaker_groups(pipeline):
    groups = {name:set(pipeline.groups[name]) for name in ("train", "val", "test")}
    if any(not group for group in groups.values()):
        raise ValueError("Train/val/test speaker groups must be nonempty")
    if any(not isinstance(s, str) or not s for group in groups.values() for s in group):
        raise ValueError("Speaker IDs must be nonempty strings")
    for left,right in (("train","val"),("train","test"),("val","test")):
        if groups[left] & groups[right]:
            raise ValueError(f"Speaker leakage between {left} and {right}")
    return groups


def role_partition(pipeline, seed=42):
    """Assign all val speakers to three nonempty roles before any scores exist."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    groups = _speaker_groups(pipeline)
    if len(groups["val"]) < 3:
        raise ValueError("At least three validation speakers are required for separate roles")
    ordered = sorted(groups["val"], key=lambda speaker:
        hashlib.sha256(f"normal-only:{seed}:{speaker}".encode()).hexdigest())
    quotient,remainder = divmod(len(ordered),3)
    roles,offset = {},0
    for index,name in enumerate(ROLE_NAMES):
        count = quotient + (index < remainder)
        roles[name] = ordered[offset:offset+count]
        offset += count
    contract = pipeline.contract()
    digest = _fingerprint(dict(seed=seed,roles=roles,data_contract=contract))
    return dict(schema_version=1, seed=seed, roles=roles, role_digest=digest,
        data_contract=contract, split="val", method="three seeded speaker-hash groups, before model/threshold selection",
        usage=dict(model_selection="normal val only for early stopping and best model selection",
                   calibration="normal val only for threshold fitting after model selection",
                   evaluation="mixed held-out val only for development metrics"),
        test_access="separate existing split, explicit opt-in only",
        limitations=["Speaker separation does not prove independent overlapping windows or absence of duplicate text across speakers.",
                     "Proxy normality is defined by next-character y, not a semantic anomaly label."])


def validate_partition(pipeline, partition):
    groups = _speaker_groups(pipeline)
    if not isinstance(partition,dict) or set(partition.get("roles",{})) != set(ROLE_NAMES):
        raise ValueError("Partition must declare the three validation roles")
    seen = set()
    for role in ROLE_NAMES:
        members = partition["roles"][role]
        if (not isinstance(members,list) or not members or len(set(members)) != len(members)
                or seen & set(members)):
            raise ValueError("Validation role speakers must be nonempty, unique and disjoint")
        seen.update(members)
    if seen != groups["val"]:
        raise ValueError("Validation roles must exhaust exactly the existing val speakers")
    if partition.get("data_contract") != pipeline.contract():
        raise ValueError("Role partition data contract differs")
    expected = _fingerprint(dict(seed=partition.get("seed"),roles=partition["roles"],data_contract=partition["data_contract"]))
    if partition.get("role_digest") != expected:
        raise ValueError("Role partition fingerprint differs")
    return True


def _row_hash(row):
    # No original data is written; hashes identify row content without text dumps.
    return _fingerprint({name:row.get(name) for name in ("character_id","x","y","y_id")})


def build_train_input_guard(pipeline, normal_class, *, deadline=None, read_batch_size=4096,
                            max_unique_inputs=1000000):
    """Hash all NORMAL train x before selecting any held-out row.

    This is a read-only quarantine view: training keeps every original normal
    row, while held-out sampling can exclude exact x already seen in training.
    Hashes are never fitted from validation/test and source files are unchanged.
    """
    _speaker_groups(pipeline)
    _normal_id(pipeline,normal_class)
    _positive(read_batch_size,"read_batch_size",8192)
    _positive(max_unique_inputs,"max_unique_inputs",1000000)
    reader=pipeline._iter_rows_batches("train",batch_size=read_batch_size)
    hashes=set()
    normal_rows=scanned=0
    ordered=hashlib.sha256()
    try:
        for rows in reader:
            _deadline(deadline)
            texts,labels=pipeline._validate_rows(rows,"train")
            scanned+=len(rows)
            for text,label in zip(texts,labels):
                if label!=normal_class:
                    continue
                digest=hashlib.sha256(text.encode("utf-8")).hexdigest()
                hashes.add(digest)
                normal_rows+=1
                ordered.update(digest.encode("ascii"))
                if len(hashes)>max_unique_inputs:
                    raise ValueError("Normal training input guard exceeded its explicit unique-input memory bound")
    finally:
        if hasattr(reader,"close"):
            reader.close()
    if not normal_rows:
        raise ValueError("No normal training inputs exist for the input-overlap guard")
    content=hashlib.sha256()
    for digest in sorted(hashes):
        content.update(digest.encode("ascii"))
    return dict(input_hashes=hashes,metadata=dict(normal_class=normal_class,
        normal_train_rows=normal_rows,scanned_train_rows=scanned,unique_inputs=len(hashes),
        input_hashes_sha256=content.hexdigest(),normal_train_input_order_sha256=ordered.hexdigest(),
        hash_algorithm="SHA-256 of exact UTF-8 x, no normalization",
        fitted_split="train_normal_only",max_unique_inputs=max_unique_inputs,
        action="exclude exact previously trained x from held-out scoring views; never modify original files",
        limitation="Exact-input quarantine does not detect paraphrases, near duplicates or semantic overlap"))


def _quotas(speakers, maximum, per_speaker_cap):
    quotient,remainder = divmod(maximum,len(speakers))
    return {speaker:min(per_speaker_cap,quotient+(i < remainder)) for i,speaker in enumerate(speakers)}


def prepare_role_samples(pipeline, partition, normal_class, *, roles=("model_selection",),
                         max_per_role=4096, per_speaker_cap=256, read_batch_size=4096,
                         deadline=None, allow_test=False, exclude_input_hashes=None):
    """Bounded deterministic reservoirs, filtering normals before selection.

    Only requested roles are retained/encoded. The reader may traverse other
    val rows as metadata while scanning the split, but never scores those rows.
    Requesting test is refused unless allow_test=True; calibration never uses it.
    """
    validate_partition(pipeline,partition)
    _normal_id(pipeline,normal_class)
    if isinstance(roles,str):
        roles = (roles,)
    roles = tuple(roles)
    if not roles or len(set(roles)) != len(roles) or set(roles) - set((*ROLE_NAMES,"test")):
        raise ValueError("Choose unique known validation roles or explicit test")
    if type(allow_test) is not bool or ("test" in roles and not allow_test):
        raise ValueError("Test sampling requires explicit allow_test=True")
    _positive(per_speaker_cap,"per_speaker_cap",MAX_SELECTED_ROWS)
    _positive(read_batch_size,"read_batch_size",8192)
    if exclude_input_hashes is not None and (not isinstance(exclude_input_hashes,(set,frozenset))
            or any(not isinstance(value,str) or len(value)!=64 for value in exclude_input_hashes)):
        raise ValueError("exclude_input_hashes must be the training guard's set of SHA-256 strings")
    if isinstance(max_per_role,dict) and any(role not in max_per_role for role in roles):
        raise ValueError("max_per_role mapping must specify every requested role")
    limits = ({role:max_per_role for role in roles} if type(max_per_role) is int else
              {role:max_per_role[role] for role in roles} if isinstance(max_per_role,dict) else None)
    if limits is None:
        raise ValueError("max_per_role must be an integer or role mapping")
    for role,limit in limits.items():
        _positive(limit,f"max_per_role[{role}]",MAX_SELECTED_ROWS)
    if sum(limits.values()) > MAX_SELECTED_ROWS:
        raise ValueError("Combined retained rows exceed the bounded protocol limit")
    samples,metadata = {},{}
    scans = {}
    for split in ("val","test"):
        selected_roles = [role for role in roles if (role == "test") == (split == "test")]
        if not selected_roles:
            continue
        members,quotas,reservoirs,seen,generators = {},{},{},{},{}
        raw_counts = {role:Counter() for role in selected_roles}
        eligible_counts = {role:Counter() for role in selected_roles}
        excluded_non_normal=Counter()
        excluded_training_overlap=Counter()
        declared = {}
        for role in selected_roles:
            speakers = (sorted(pipeline.groups["test"]) if role == "test" else partition["roles"][role])
            declared[role] = speakers
            role_quotas = _quotas(speakers,limits[role],per_speaker_cap)
            for speaker in speakers:
                key = (role,speaker)
                members[speaker] = role
                quotas[key],reservoirs[key],seen[key] = role_quotas[speaker],[],0
                seed = int.from_bytes(hashlib.sha256(f"{partition['seed']}:{split}:{role}:{speaker}".encode()).digest()[:8],"little")
                generators[key] = random.Random(seed)
        reader = pipeline._iter_rows_batches(split,batch_size=read_batch_size)
        scanned = 0
        try:
            for rows in reader:
                _deadline(deadline)
                _,labels = pipeline._validate_rows(rows,split)
                for row,label in zip(rows,labels):
                    index = scanned
                    scanned += 1
                    role = members.get(row["character_id"])
                    if role is None:
                        continue
                    raw_counts[role][label] += 1
                    if role in ("model_selection","calibration") and label != normal_class:
                        excluded_non_normal[role]+=1
                        continue
                    input_hash=None
                    if exclude_input_hashes is not None:
                        input_hash=hashlib.sha256(row["x"].encode("utf-8")).hexdigest()
                        if input_hash in exclude_input_hashes:
                            excluded_training_overlap[role]+=1
                            continue
                    eligible_counts[role][label] += 1
                    key = (role,row["character_id"])
                    seen[key] += 1
                    capacity = quotas[key]
                    if not capacity:
                        continue
                    entry = dict(row=dict(row),row_index=index,label=label)
                    if input_hash is not None:
                        entry["input_sha256"]=input_hash
                    if len(reservoirs[key]) < capacity:
                        reservoirs[key].append(entry)
                    else:
                        replacement = generators[key].randrange(seen[key])
                        if replacement < capacity:
                            reservoirs[key][replacement] = entry
        finally:
            if hasattr(reader,"close"):
                reader.close()
        scans[split] = scanned
        for role in selected_roles:
            entries = sorted([entry for speaker in declared[role] for entry in reservoirs[role,speaker]],
                             key=lambda entry:entry["row_index"])
            for entry in entries:
                entry["row_sha256"] = _row_hash(entry["row"])
                entry["input_sha256"] = entry.get("input_sha256") or hashlib.sha256(entry["row"]["x"].encode("utf-8")).hexdigest()
            counts = Counter(entry["label"] for entry in entries)
            samples[role] = entries
            fingerprint = _fingerprint([[e["row_index"],e["row_sha256"]] for e in entries])
            normal_count = counts[normal_class]
            metadata[role] = dict(split=split,normal_only=role in ("model_selection","calibration"),
                selected_rows=len(entries),normal_rows=normal_count,anomaly_rows=len(entries)-normal_count,
                anomaly_prevalence=(len(entries)-normal_count)/len(entries) if entries else None,
                raw_rows_in_role=sum(raw_counts[role].values()),eligible_rows=sum(eligible_counts[role].values()),
                excluded_non_normal_rows=excluded_non_normal[role],
                excluded_training_input_overlap_rows=excluded_training_overlap[role],
                raw_label_counts={str(k):v for k,v in sorted(raw_counts[role].items())},
                selected_label_counts={str(k):v for k,v in sorted(counts.items())},
                requested_maximum=limits[role],speakers_declared=len(declared[role]),
                speakers_with_samples=sum(bool(reservoirs[role,s]) for s in declared[role]),
                per_speaker={s:dict(quota=quotas[role,s],eligible=seen[role,s],selected=len(reservoirs[role,s])) for s in declared[role]},
                rows_sha256=fingerprint,population_fingerprint=fingerprint)
    sampling = dict(role_digest=partition["role_digest"],roles=metadata,scanned_rows=scans,
        training_input_guard_applied=exclude_input_hashes is not None,
        training_input_hash_count=len(exclude_input_hashes) if exclude_input_hashes is not None else None,
        excluded_training_input_overlap_rows_total=sum(m["excluded_training_input_overlap_rows"] for m in metadata.values()),
        method="full requested-split scan with bounded seeded reservoirs per requested role/speaker",
        score_access="none; sampling and label bookkeeping only",
        population_fingerprint=_fingerprint({role:metadata[role]["rows_sha256"] for role in sorted(metadata)}),
        limitations=["Speaker-stratified subsets do not preserve population class prevalence exactly.",
                     "No claim of complete cross-speaker exact-text deduplication or IID windows."])
    return dict(samples=samples,sampling=sampling)


def role_tensor(pipeline, entries, split="val"):
    if not entries:
        raise ValueError("Requested role contains no retained examples")
    return pipeline.encode_tensor_rows([entry["row"] for entry in entries],split)


def iter_normal_train_batches(pipeline, normal_class, *, batch_size=128, epoch=1, seed=42,
                              shuffle_buffer_batches=16, read_batch_size=4096, deadline=None):
    """Yield every normal train row exactly once per epoch with bounded shuffle.

    Filtering by y occurs BEFORE filling the shuffle buffer/batches. The final
    partial batch is emitted; no over/undersampling or anomalous training rows.
    Local RNG does not consume global Python/Torch RNG state.
    """
    _speaker_groups(pipeline)
    _normal_id(pipeline,normal_class)
    _positive(batch_size,"batch_size",4096)
    _positive(epoch,"epoch",1000000)
    _positive(shuffle_buffer_batches,"shuffle_buffer_batches",128)
    _positive(read_batch_size,"read_batch_size",8192)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    capacity = batch_size*shuffle_buffer_batches
    if capacity > MAX_SELECTED_ROWS:
        raise ValueError("Training shuffle buffer exceeds its memory bound")
    rng = random.Random((seed+epoch) % 2**32)
    reader = pipeline._iter_rows_batches("train",batch_size=read_batch_size)
    buffer,index = [],0
    def batches():
        rng.shuffle(buffer)
        for start in range(0,len(buffer),batch_size):
            _deadline(deadline)
            part = buffer[start:start+batch_size]
            encoded = pipeline.encode_tensor_rows([entry[1] for entry in part],"train")
            if (encoded["y"] != normal_class).any().item():
                raise RuntimeError("Anomalous label reached normal-only training batch")
            yield {**encoded,"row_indices":[entry[0] for entry in part],
                   "character_ids":[entry[1]["character_id"] for entry in part],"split":"train"}
    try:
        for rows in reader:
            _deadline(deadline)
            _,labels = pipeline._validate_rows(rows,"train")
            for row,label in zip(rows,labels):
                current,index = index,index+1
                if label != normal_class:
                    continue
                buffer.append((current,row))
                if len(buffer) == capacity:
                    yield from batches()
                    buffer.clear()
        if buffer:
            yield from batches()
    finally:
        if hasattr(reader,"close"):
            reader.close()
