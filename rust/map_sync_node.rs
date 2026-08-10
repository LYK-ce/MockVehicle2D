use std::{
    collections::{HashMap, HashSet, hash_map::DefaultHasher},
    env,
    error::Error,
    ffi::{CString, OsString},
    fs::{self, File, OpenOptions},
    hash::{Hash, Hasher},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, FromRawFd},
        unix::{
            ffi::OsStrExt,
            fs::{MetadataExt, OpenOptionsExt, PermissionsExt},
        },
    },
    path::{Path, PathBuf},
    process,
    str::FromStr,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use futures::StreamExt;
use libp2p::{
    Multiaddr, PeerId, SwarmBuilder, gossipsub, identity, noise,
    swarm::{NetworkBehaviour, SwarmEvent},
    tcp, yamux,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{UnixStream, unix::OwnedWriteHalf},
    select, time,
};

const SIDECAR_PROTOCOL: &str = "mockvehicle2d-map-sync-sidecar/1";
const DELTA_PROTOCOL: &str = "mockvehicle2d-map-delta/1";
const PEER_STATE_PROTOCOL: &str = "mockvehicle2d-peer-state/1";
const MOTION_INTENT_PROTOCOL: &str = "mockvehicle2d-motion-intent/4";
const MOTION_COMMIT_HORIZON_S: f64 = 0.8;
const MAX_MESSAGE_BYTES: usize = 256 * 1024;
const MAX_GRID_COORDINATE: i64 = 1_000_000;
const MAX_PEERS: usize = 3;
static IDENTITY_TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

type BoxError = Box<dyn Error + Send + Sync>;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct PeerConfig {
    vehicle_id: String,
    peer_id: String,
    address: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeConfig {
    protocol: String,
    vehicle_id: String,
    session_id: String,
    listen_port: u16,
    uds_path: PathBuf,
    identity_path: PathBuf,
    peers: Vec<PeerConfig>,
}

#[derive(Debug)]
struct KnownPeer {
    vehicle_id: String,
    peer_id: PeerId,
    address: Multiaddr,
}

#[derive(NetworkBehaviour)]
struct Behaviour {
    gossipsub: gossipsub::Behaviour,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum LocalCommand {
    Publish { payload: Value },
    Shutdown,
}

#[derive(Serialize)]
struct ReadyEvent<'a> {
    r#type: &'static str,
    protocol: &'static str,
    vehicle_id: &'a str,
    peer_id: String,
    listen_port: u16,
}

#[derive(Serialize)]
struct PeerHealthEvent<'a> {
    r#type: &'static str,
    vehicle_id: &'a str,
    connected_vehicle_ids: Vec<&'a str>,
}

#[derive(Serialize)]
struct ReceivedEvent<'a> {
    r#type: &'static str,
    source_peer_id: String,
    source_vehicle_id: &'a str,
    payload: Value,
}

#[derive(Serialize)]
struct PublishResult {
    r#type: &'static str,
    protocol: String,
    sequence: u64,
    accepted: bool,
    error: Option<String>,
}

fn valid_identifier(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

impl NodeConfig {
    fn load(path: &Path) -> Result<Self, BoxError> {
        let raw = fs::read(path)?;
        if raw.len() > MAX_MESSAGE_BYTES {
            return Err("sidecar config is too large".into());
        }
        let config: Self = serde_json::from_slice(&raw)?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), BoxError> {
        if self.protocol != SIDECAR_PROTOCOL {
            return Err("unsupported sidecar protocol".into());
        }
        if !valid_identifier(&self.vehicle_id) || !valid_identifier(&self.session_id) {
            return Err("invalid vehicle_id or session_id".into());
        }
        if self.listen_port == 0 || self.uds_path.as_os_str().is_empty() {
            return Err("listen_port and uds_path are required".into());
        }
        if self.peers.len() > MAX_PEERS {
            return Err("at most three remote peers are supported".into());
        }
        let mut vehicle_ids = HashSet::new();
        let mut peer_ids = HashSet::new();
        for peer in &self.peers {
            if !valid_identifier(&peer.vehicle_id) || peer.vehicle_id == self.vehicle_id {
                return Err("invalid remote vehicle_id".into());
            }
            let peer_id = PeerId::from_str(&peer.peer_id)?;
            let address = Multiaddr::from_str(&peer.address)?;
            if !vehicle_ids.insert(peer.vehicle_id.as_str())
                || !peer_ids.insert(peer_id)
                || address.is_empty()
            {
                return Err("peer identities and addresses must be unique".into());
            }
        }
        Ok(())
    }

    fn known_peers(&self) -> Result<Vec<KnownPeer>, BoxError> {
        self.peers
            .iter()
            .map(|peer| {
                Ok(KnownPeer {
                    vehicle_id: peer.vehicle_id.clone(),
                    peer_id: PeerId::from_str(&peer.peer_id)?,
                    address: Multiaddr::from_str(&peer.address)?,
                })
            })
            .collect()
    }
}

fn identity_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn effective_user_id() -> u32 {
    // SAFETY: geteuid has no preconditions and does not dereference pointers.
    unsafe { libc::geteuid() }
}

fn secure_identity_parent(path: &Path, create: bool) -> Result<File, BoxError> {
    let parent = identity_parent(path);
    if create {
        fs::create_dir_all(parent)?;
    }
    let directory = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
        .open(parent)
        .map_err(|error| {
            io::Error::new(
                error.kind(),
                format!("identity parent directory is unsafe: {error}"),
            )
        })?;
    let metadata = directory.metadata()?;
    if !metadata.is_dir() || metadata.uid() != effective_user_id() || metadata.mode() & 0o022 != 0 {
        return Err(format!(
            "identity parent directory is unsafe: it must be owned by this user and not group/world writable (uid={}, expected_uid={}, mode={:o})",
            metadata.uid(),
            effective_user_id(),
            metadata.mode() & 0o777,
        )
        .into());
    }
    Ok(directory)
}

fn identity_file_name(path: &Path) -> Result<CString, BoxError> {
    let name = path.file_name().ok_or("identity path must name a file")?;
    CString::new(name.as_bytes()).map_err(|_| "identity filename contains a NUL byte".into())
}

fn open_identity_at(directory: &File, name: &CString, flags: i32, mode: u32) -> io::Result<File> {
    // SAFETY: directory and name remain alive for the call; openat returns a new owned fd.
    let fd = unsafe { libc::openat(directory.as_raw_fd(), name.as_ptr(), flags, mode) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: the successful openat result is a new descriptor now owned by File.
    Ok(unsafe { File::from_raw_fd(fd) })
}

fn unlink_identity_at(directory: &File, name: &CString) -> io::Result<()> {
    // SAFETY: directory and name remain alive and name is relative to the held directory fd.
    if unsafe { libc::unlinkat(directory.as_raw_fd(), name.as_ptr(), 0) } == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

fn existing_identity(path: &Path) -> Result<Option<identity::Keypair>, BoxError> {
    let parent = secure_identity_parent(path, false)?;
    let name = identity_file_name(path)?;
    let mut file = match open_identity_at(
        &parent,
        &name,
        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        0,
    ) {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) if error.raw_os_error() == Some(libc::ELOOP) => {
            return Err("existing identity path is a symlink; refusing to follow it".into());
        }
        Err(error) => return Err(error.into()),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.uid() != effective_user_id() {
        return Err("existing identity path is not a regular file; refusing to replace".into());
    }
    file.set_permissions(fs::Permissions::from_mode(0o600))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    identity::Keypair::from_protobuf_encoding(&bytes)
        .map(Some)
        .map_err(|error| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("existing identity file is invalid; refusing to replace: {error}"),
            )
            .into()
        })
}

fn create_identity_temporary(
    path: &Path,
    directory: &File,
) -> Result<(PathBuf, CString, File), BoxError> {
    let parent = identity_parent(path);
    let file_name = path.file_name().ok_or("identity path must name a file")?;
    for _ in 0..100 {
        let mut temporary_name = OsString::from(".");
        temporary_name.push(file_name);
        temporary_name.push(format!(
            ".tmp.{}.{}",
            process::id(),
            IDENTITY_TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let temporary_path = parent.join(temporary_name);
        let temporary_name = identity_file_name(&temporary_path)?;
        match open_identity_at(
            directory,
            &temporary_name,
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            0o600,
        ) {
            Ok(file) => {
                if let Err(error) = file.set_permissions(fs::Permissions::from_mode(0o600)) {
                    drop(file);
                    return match unlink_identity_at(directory, &temporary_name) {
                        Ok(()) => Err(error.into()),
                        Err(cleanup_error) => Err(io::Error::other(format!(
                            "cannot secure identity temporary ({error}) and cannot remove it: \
                             {cleanup_error}"
                        ))
                        .into()),
                    };
                }
                return Ok((temporary_path, temporary_name, file));
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Err("cannot create a unique identity temporary file".into())
}

fn write_identity_atomically_with<F>(
    path: &Path,
    bytes: &[u8],
    before_publish: F,
) -> Result<bool, BoxError>
where
    F: FnOnce(&Path) -> io::Result<()>,
{
    let directory = secure_identity_parent(path, true)?;
    let final_name = identity_file_name(path)?;
    let (temporary_path, temporary_name, mut file) = create_identity_temporary(path, &directory)?;
    let result = (|| -> Result<bool, BoxError> {
        file.write_all(bytes)?;
        file.sync_all()?;
        before_publish(&temporary_path)?;
        drop(file);
        // SAFETY: both names are relative to the same held directory fd; linkat never replaces.
        let linked = unsafe {
            libc::linkat(
                directory.as_raw_fd(),
                temporary_name.as_ptr(),
                directory.as_raw_fd(),
                final_name.as_ptr(),
                0,
            )
        };
        let published = if linked == 0 {
            true
        } else {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::AlreadyExists {
                false
            } else {
                return Err(error.into());
            }
        };
        unlink_identity_at(&directory, &temporary_name)?;
        directory.sync_all()?;
        Ok(published)
    })();
    if let Err(original) = &result {
        let original_error = original.to_string();
        match unlink_identity_at(&directory, &temporary_name) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(cleanup_error) => {
                return Err(io::Error::other(format!(
                    "identity write failed ({original_error}) and temporary cleanup failed: \
                     {cleanup_error}"
                ))
                .into());
            }
        }
    }
    result
}

fn load_or_create_identity_with<F>(
    path: &Path,
    before_publish: F,
) -> Result<identity::Keypair, BoxError>
where
    F: FnOnce(&Path) -> io::Result<()>,
{
    secure_identity_parent(path, true)?;
    if let Some(key) = existing_identity(path)? {
        return Ok(key);
    }

    let key = identity::Keypair::generate_ed25519();
    let bytes = key.to_protobuf_encoding()?;
    if write_identity_atomically_with(path, &bytes, before_publish)? {
        return Ok(key);
    }
    existing_identity(path)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "identity was concurrently published but is no longer present",
        )
        .into()
    })
}

fn load_or_create_identity(path: &Path) -> Result<identity::Keypair, BoxError> {
    load_or_create_identity_with(path, |_| Ok(()))
}

async fn send_event(writer: &mut OwnedWriteHalf, event: &impl Serialize) -> Result<(), BoxError> {
    let mut encoded = serde_json::to_vec(event)?;
    encoded.push(b'\n');
    writer.write_all(&encoded).await?;
    Ok(())
}

fn payload_identity(payload: &Value) -> Result<(&str, &str, &str, u64), &'static str> {
    let object = payload.as_object().ok_or("payload must be an object")?;
    let protocol = object
        .get("protocol")
        .and_then(Value::as_str)
        .ok_or("missing protocol")?;
    if !matches!(
        protocol,
        DELTA_PROTOCOL | PEER_STATE_PROTOCOL | MOTION_INTENT_PROTOCOL
    ) {
        return Err("unsupported map-sync protocol");
    }
    let session = object
        .get("session_id")
        .and_then(Value::as_str)
        .ok_or("missing session_id")?;
    let vehicle = object
        .get("source_vehicle_id")
        .and_then(Value::as_str)
        .ok_or("missing source_vehicle_id")?;
    let sequence = object
        .get("sequence")
        .and_then(Value::as_u64)
        .ok_or("missing sequence")?;
    if protocol == MOTION_INTENT_PROTOCOL {
        validate_motion_intent_timing(object)?;
    }
    Ok((protocol, session, vehicle, sequence))
}

fn validate_motion_intent_timing(
    payload: &serde_json::Map<String, Value>,
) -> Result<(), &'static str> {
    let current_cell = valid_cell(payload.get("current_cell")).ok_or("current_cell is invalid")?;
    validate_vacate_request(payload.get("vacate_request"), current_cell)?;
    let committed = payload
        .get("committed_until_offset_s")
        .and_then(Value::as_f64)
        .ok_or("missing committed_until_offset_s")?;
    if !(0.0..=MOTION_COMMIT_HORIZON_S).contains(&committed) {
        return Err("motion commit exceeds the supported horizon");
    }
    let trajectory = payload
        .get("trajectory")
        .and_then(Value::as_array)
        .filter(|trajectory| !trajectory.is_empty())
        .ok_or("missing trajectory")?;
    let mut previous: Option<(&Value, f64)> = None;
    for item in trajectory {
        let item = item
            .as_object()
            .ok_or("trajectory item must be an object")?;
        let cell = item.get("cell").ok_or("trajectory cell is missing")?;
        let enter = item
            .get("enter_offset_s")
            .and_then(Value::as_f64)
            .ok_or("trajectory enter time is missing")?;
        let leave = item
            .get("leave_offset_s")
            .and_then(Value::as_f64)
            .ok_or("trajectory leave time is missing")?;
        if enter < 0.0 || leave < enter {
            return Err("trajectory interval is invalid");
        }
        if let Some((previous_cell, previous_leave)) = previous
            && (cell == previous_cell || enter <= previous_leave)
        {
            return Err("trajectory travel time must be positive");
        }
        previous = Some((cell, leave));
    }
    if committed
        > previous
            .map(|(_, leave)| leave)
            .ok_or("missing trajectory")?
    {
        return Err("motion commit exceeds trajectory");
    }
    Ok(())
}

fn valid_cell(value: Option<&Value>) -> Option<(i64, i64)> {
    let cell = value?.as_object()?;
    if cell.len() != 2 {
        return None;
    }
    let coordinate = |name| {
        cell.get(name)
            .and_then(Value::as_i64)
            .filter(|value| (-MAX_GRID_COORDINATE..=MAX_GRID_COORDINATE).contains(value))
    };
    Some((coordinate("gx")?, coordinate("gy")?))
}

fn validate_vacate_request(
    value: Option<&Value>,
    current_cell: (i64, i64),
) -> Result<(), &'static str> {
    match value {
        Some(Value::Null) => Ok(()),
        Some(Value::Object(request)) => {
            request
                .get("vehicle_id")
                .and_then(Value::as_str)
                .filter(|vehicle_id| valid_identifier(vehicle_id))
                .ok_or("vacate_request vehicle_id is invalid")?;
            valid_cell(request.get("cell")).ok_or("vacate_request cell is invalid")?;
            let route_cells = request
                .get("route_cells")
                .and_then(Value::as_array)
                .ok_or("vacate_request route_cells is invalid")?
                .iter()
                .map(|cell| valid_cell(Some(cell)))
                .collect::<Option<Vec<_>>>()
                .ok_or("vacate_request route cell is invalid")?;
            if request.len() != 3
                || !(2..=64).contains(&route_cells.len())
                || route_cells.first() != Some(&current_cell)
                || route_cells.windows(2).any(|cells| {
                    cells[0] == cells[1]
                        || cells[0].0.abs_diff(cells[1].0) > 2
                        || cells[0].1.abs_diff(cells[1].1) > 2
                })
            {
                return Err("vacate_request must be an exact bounded object");
            }
            Ok(())
        }
        _ => Err("vacate_request must be null or an exact object"),
    }
}

fn authorized_payload<'a>(
    payload: &Value,
    source: &PeerId,
    session_id: &str,
    peer_by_id: &'a HashMap<PeerId, String>,
) -> Option<(&'a str, u64)> {
    let (_, session, vehicle, sequence) = payload_identity(payload).ok()?;
    let expected_vehicle = peer_by_id.get(source)?;
    (session == session_id && vehicle == expected_vehicle)
        .then_some((expected_vehicle.as_str(), sequence))
}

async fn run(config: NodeConfig) -> Result<(), BoxError> {
    let key = load_or_create_identity(&config.identity_path)?;
    let local_peer_id = key.public().to_peer_id();
    let known_peers = config.known_peers()?;
    let peer_by_id: HashMap<PeerId, String> = known_peers
        .iter()
        .map(|peer| (peer.peer_id, peer.vehicle_id.clone()))
        .collect();

    let message_id_fn = |message: &gossipsub::Message| {
        let mut hasher = DefaultHasher::new();
        message.data.hash(&mut hasher);
        gossipsub::MessageId::from(hasher.finish().to_string())
    };
    let gossipsub_config = gossipsub::ConfigBuilder::default()
        .heartbeat_interval(Duration::from_millis(100))
        .validation_mode(gossipsub::ValidationMode::Strict)
        .max_transmit_size(MAX_MESSAGE_BYTES)
        .message_id_fn(message_id_fn)
        .build()
        .map_err(io::Error::other)?;
    let behaviour = Behaviour {
        gossipsub: gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(key.clone()),
            gossipsub_config,
        )?,
    };
    let mut swarm = SwarmBuilder::with_existing_identity(key)
        .with_tokio()
        .with_tcp(
            tcp::Config::default().nodelay(true),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_behaviour(|_| behaviour)?
        .build();
    let topic =
        gossipsub::IdentTopic::new(format!("mockvehicle2d/{}/fleet-sync/1", config.session_id));
    swarm.behaviour_mut().gossipsub.subscribe(&topic)?;
    for peer in &known_peers {
        swarm
            .behaviour_mut()
            .gossipsub
            .add_explicit_peer(&peer.peer_id);
    }
    swarm.listen_on(format!("/ip4/127.0.0.1/tcp/{}", config.listen_port).parse()?)?;

    let uds = time::timeout(Duration::from_secs(10), async {
        loop {
            match UnixStream::connect(&config.uds_path).await {
                Ok(stream) => return Ok::<UnixStream, io::Error>(stream),
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
                    ) =>
                {
                    time::sleep(Duration::from_millis(50)).await;
                }
                Err(error) => return Err(error),
            }
        }
    })
    .await
    .map_err(|_| "timed out connecting to the local map-sync socket")??;
    let (reader, mut writer) = uds.into_split();
    let mut lines = BufReader::new(reader).lines();
    let mut dial_tick = time::interval(Duration::from_millis(250));
    let mut connected = HashSet::new();
    let mut announced_ready = false;

    loop {
        select! {
            line = lines.next_line() => {
                let Some(line) = line? else { break };
                if line.len() > MAX_MESSAGE_BYTES {
                    return Err("local command exceeds size limit".into());
                }
                match serde_json::from_str::<LocalCommand>(&line)? {
                    LocalCommand::Shutdown => break,
                    LocalCommand::Publish { payload } => {
                        let identity = payload_identity(&payload);
                        let protocol = identity
                            .as_ref()
                            .map(|identity| identity.0)
                            .unwrap_or("");
                        let sequence = identity.as_ref().map(|identity| identity.3).unwrap_or(0);
                        let encoded = serde_json::to_vec(&payload)?;
                        let result = if identity
                            .as_ref()
                            .map(|(_, session, vehicle, _)| {
                                *session != config.session_id || *vehicle != config.vehicle_id
                            })
                            .unwrap_or(true)
                        {
                            Err("local payload identity does not match the sidecar".to_string())
                        } else if encoded.len() > MAX_MESSAGE_BYTES {
                            Err("map-sync payload exceeds size limit".to_string())
                        } else {
                            swarm
                                .behaviour_mut()
                                .gossipsub
                                .publish(topic.clone(), encoded)
                                .map(|_| ())
                                .map_err(|error| error.to_string())
                        };
                        send_event(&mut writer, &PublishResult {
                            r#type: "publish_result",
                            protocol: protocol.to_string(),
                            sequence,
                            accepted: result.is_ok(),
                            error: result.err(),
                        }).await?;
                    }
                }
            }
            _ = dial_tick.tick() => {
                for peer in &known_peers {
                    if !swarm.is_connected(&peer.peer_id) {
                        let _ = swarm.dial(peer.address.clone());
                    }
                }
            }
            event = swarm.select_next_some() => match event {
                SwarmEvent::NewListenAddr { .. } if !announced_ready => {
                    announced_ready = true;
                    send_event(&mut writer, &ReadyEvent {
                        r#type: "ready",
                        protocol: SIDECAR_PROTOCOL,
                        vehicle_id: &config.vehicle_id,
                        peer_id: local_peer_id.to_string(),
                        listen_port: config.listen_port,
                    }).await?;
                }
                SwarmEvent::ConnectionEstablished { peer_id, .. }
                    if peer_by_id.contains_key(&peer_id) && connected.insert(peer_id) => {
                    let mut connected_vehicle_ids: Vec<_> = connected
                        .iter()
                        .filter_map(|peer| peer_by_id.get(peer).map(String::as_str))
                        .collect();
                    connected_vehicle_ids.sort_unstable();
                    send_event(&mut writer, &PeerHealthEvent {
                        r#type: "peer_health",
                        vehicle_id: &config.vehicle_id,
                        connected_vehicle_ids,
                    }).await?;
                }
                SwarmEvent::ConnectionClosed { peer_id, num_established: 0, .. }
                    if connected.remove(&peer_id) => {
                    let mut connected_vehicle_ids: Vec<_> = connected
                        .iter()
                        .filter_map(|peer| peer_by_id.get(peer).map(String::as_str))
                        .collect();
                    connected_vehicle_ids.sort_unstable();
                    send_event(&mut writer, &PeerHealthEvent {
                        r#type: "peer_health",
                        vehicle_id: &config.vehicle_id,
                        connected_vehicle_ids,
                    }).await?;
                }
                SwarmEvent::Behaviour(BehaviourEvent::Gossipsub(gossipsub::Event::Message {
                    message,
                    ..
                })) => {
                    if message.data.len() > MAX_MESSAGE_BYTES {
                        continue;
                    }
                    let Some(source) = message.source else { continue };
                    let Ok(payload) = serde_json::from_slice::<Value>(&message.data) else { continue };
                    let Some((source_vehicle_id, _)) = authorized_payload(
                        &payload,
                        &source,
                        &config.session_id,
                        &peer_by_id,
                    ) else { continue };
                    send_event(&mut writer, &ReceivedEvent {
                        r#type: "received",
                        source_peer_id: source.to_string(),
                        source_vehicle_id,
                        payload,
                    }).await?;
                }
                _ => {}
            },
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    Ok(())
}

fn usage() -> &'static str {
    "usage: map-sync-node identity IDENTITY_PATH | map-sync-node run CONFIG_JSON"
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let mut args = env::args_os().skip(1);
    match (args.next(), args.next(), args.next()) {
        (Some(command), Some(path), None) if command == "identity" => {
            let key = load_or_create_identity(Path::new(&path))?;
            println!("{}", key.public().to_peer_id());
            Ok(())
        }
        (Some(command), Some(path), None) if command == "run" => {
            run(NodeConfig::load(Path::new(&path))?).await
        }
        _ => Err(usage().into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;

    fn private_tempdir() -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        directory
    }

    fn motion_intent_v4(vacate_request: Value) -> Value {
        serde_json::json!({
            "protocol": MOTION_INTENT_PROTOCOL,
            "session_id": "session_1",
            "source_vehicle_id": "vehicle_1",
            "sequence": 1,
            "current_cell": {"gx": 0, "gy": 0},
            "trajectory": [{
                "cell": {"gx": 0, "gy": 0},
                "enter_offset_s": 0.0,
                "leave_offset_s": 4.0
            }],
            "committed_until_offset_s": 0.8,
            "vacate_request": vacate_request
        })
    }

    #[test]
    fn identity_is_stable_and_unique() {
        let directory = private_tempdir();
        let first_path = directory.path().join("first.key");
        let second_path = directory.path().join("second.key");
        let first = load_or_create_identity(&first_path)
            .unwrap()
            .public()
            .to_peer_id();
        let repeated = load_or_create_identity(&first_path)
            .unwrap()
            .public()
            .to_peer_id();
        let second = load_or_create_identity(&second_path)
            .unwrap()
            .public()
            .to_peer_id();

        assert_eq!(first, repeated);
        assert_ne!(first, second);
        assert_eq!(
            fs::metadata(first_path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn concurrent_identity_creation_returns_the_single_published_identity() {
        let directory = private_tempdir();
        let path = Arc::new(directory.path().join("identity.key"));
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let path = Arc::clone(&path);
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    load_or_create_identity_with(&path, |_| {
                        barrier.wait();
                        Ok(())
                    })
                    .unwrap()
                    .public()
                    .to_peer_id()
                })
            })
            .collect();
        let peers: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        assert!(peers.iter().all(|peer| peer == &peers[0]));
        assert_eq!(
            load_or_create_identity(&path)
                .unwrap()
                .public()
                .to_peer_id(),
            peers[0]
        );
        assert_eq!(
            fs::metadata(&*path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::read_dir(directory.path()).unwrap().count(),
            1,
            "concurrent creation must clean every private temporary"
        );
    }

    #[test]
    fn identity_rejects_symlink_and_unsafe_parent() {
        use std::os::unix::fs::symlink;

        let directory = private_tempdir();
        let victim = directory.path().join("victim");
        fs::write(&victim, b"do not touch").unwrap();
        let link = directory.path().join("identity.key");
        symlink(&victim, &link).unwrap();

        let error = load_or_create_identity(&link).unwrap_err().to_string();
        assert!(error.contains("symlink") || error.contains("regular file"));
        assert_eq!(fs::read(&victim).unwrap(), b"do not touch");

        let unsafe_parent = directory.path().join("unsafe");
        fs::create_dir(&unsafe_parent).unwrap();
        fs::set_permissions(&unsafe_parent, fs::Permissions::from_mode(0o777)).unwrap();
        let error = load_or_create_identity(&unsafe_parent.join("identity.key"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("identity parent directory is unsafe"));
        assert!(!unsafe_parent.join("identity.key").exists());

        let private_parent = directory.path().join("private");
        fs::create_dir(&private_parent).unwrap();
        fs::set_permissions(&private_parent, fs::Permissions::from_mode(0o700)).unwrap();
        let parent_link = directory.path().join("parent-link");
        symlink(&private_parent, &parent_link).unwrap();
        let error = load_or_create_identity(&parent_link.join("identity.key"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("identity parent directory is unsafe"));
        assert!(!private_parent.join("identity.key").exists());
    }

    #[test]
    fn corrupt_identity_is_not_silently_replaced() {
        let directory = private_tempdir();
        let path = directory.path().join("identity.key");
        let corrupt = b"not a protobuf identity";
        fs::write(&path, corrupt).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o666)).unwrap();

        let error = load_or_create_identity(&path).unwrap_err().to_string();

        assert!(error.contains("invalid; refusing to replace"));
        assert_eq!(fs::read(&path).unwrap(), corrupt);
        assert_eq!(
            fs::metadata(path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }

    #[test]
    fn failed_identity_publish_preserves_final_and_cleans_private_temporary() {
        let directory = private_tempdir();
        let path = directory.path().join("identity.key");
        let original = identity::Keypair::generate_ed25519()
            .to_protobuf_encoding()
            .unwrap();
        let replacement = identity::Keypair::generate_ed25519()
            .to_protobuf_encoding()
            .unwrap();
        fs::write(&path, &original).unwrap();

        let error = write_identity_atomically_with(&path, &replacement, |temporary| {
            assert_eq!(
                fs::metadata(temporary).unwrap().permissions().mode() & 0o777,
                0o600
            );
            assert_eq!(fs::read(temporary).unwrap(), replacement);
            Err(io::Error::other("injected before identity rename"))
        })
        .unwrap_err()
        .to_string();

        assert!(error.contains("injected before identity rename"));
        assert_eq!(fs::read(&path).unwrap(), original);
        let entries: Vec<_> = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(entries, vec![OsString::from("identity.key")]);
    }

    #[test]
    fn authorization_binds_signed_peer_to_vehicle_and_session() {
        let peer = identity::Keypair::generate_ed25519().public().to_peer_id();
        let peers = HashMap::from([(peer, "vehicle_1".to_string())]);
        for protocol in [PEER_STATE_PROTOCOL, MOTION_INTENT_PROTOCOL] {
            let mut payload = serde_json::json!({
                "protocol": protocol,
                "session_id": "session_1",
                "source_vehicle_id": "vehicle_1",
                "sequence": 1
            });
            if protocol == MOTION_INTENT_PROTOCOL {
                payload["current_cell"] = serde_json::json!({"gx": 0, "gy": 0});
                payload["vacate_request"] = Value::Null;
                payload["trajectory"] = serde_json::json!([{
                    "cell": {"gx": 0, "gy": 0},
                    "enter_offset_s": 0.0,
                    "leave_offset_s": 4.0
                }]);
                payload["committed_until_offset_s"] = serde_json::json!(0.8);
            }

            assert_eq!(
                authorized_payload(&payload, &peer, "session_1", &peers),
                Some(("vehicle_1", 1)),
            );
            assert!(authorized_payload(&payload, &peer, "other", &peers).is_none());
        }
    }

    #[test]
    fn motion_intent_v4_rejects_teleport_and_commit_past_short_horizon() {
        let valid = serde_json::json!({
            "protocol": MOTION_INTENT_PROTOCOL,
            "session_id": "session_1",
            "source_vehicle_id": "vehicle_1",
            "sequence": 1,
            "current_cell": {"gx": 0, "gy": 0},
            "trajectory": [
                {
                    "cell": {"gx": 0, "gy": 0},
                    "enter_offset_s": 0.0,
                    "leave_offset_s": 0.0
                },
                {
                    "cell": {"gx": 1, "gy": 0},
                    "enter_offset_s": 0.5,
                    "leave_offset_s": 4.0
                }
            ],
            "committed_until_offset_s": 0.8,
            "vacate_request": null
        });
        assert!(payload_identity(&valid).is_ok());

        let mut teleport = valid.clone();
        teleport["trajectory"][1]["enter_offset_s"] = serde_json::json!(0.0);
        assert!(payload_identity(&teleport).is_err());

        let mut excessive_commit = valid.clone();
        excessive_commit["committed_until_offset_s"] = serde_json::json!(0.8000000000000002);
        assert!(payload_identity(&excessive_commit).is_err());

        let mut hold = valid;
        hold["trajectory"] = serde_json::json!([{
            "cell": {"gx": 0, "gy": 0},
            "enter_offset_s": 0.0,
            "leave_offset_s": 0.1
        }]);
        assert!(payload_identity(&hold).is_err());
        hold["committed_until_offset_s"] = serde_json::json!(0.1);
        assert!(payload_identity(&hold).is_ok());
    }

    #[test]
    fn motion_intent_v4_requires_vacate_request_field() {
        assert_eq!(MOTION_INTENT_PROTOCOL, "mockvehicle2d-motion-intent/4");
        let mut payload = motion_intent_v4(Value::Null);
        assert!(payload_identity(&payload).is_ok());

        let mut old = payload.clone();
        old["protocol"] = serde_json::json!("mockvehicle2d-motion-intent/3");
        assert!(payload_identity(&old).is_err());

        payload.as_object_mut().unwrap().remove("vacate_request");
        assert!(payload_identity(&payload).is_err());
    }

    #[test]
    fn motion_intent_v4_vacate_request_has_exact_fields() {
        let valid = motion_intent_v4(serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1, "gy": 0},
                "route_cells": [
                    {"gx": 0, "gy": 0},
                    {"gx": 1, "gy": 0},
                    {"gx": 3, "gy": 0}
                ]
        }));
        assert!(payload_identity(&valid).is_ok());

        let mut missing = valid.clone();
        missing["vacate_request"]
            .as_object_mut()
            .unwrap()
            .remove("cell");
        assert!(payload_identity(&missing).is_err());

        let mut missing_route = valid.clone();
        missing_route["vacate_request"]
            .as_object_mut()
            .unwrap()
            .remove("route_cells");
        assert!(payload_identity(&missing_route).is_err());

        let mut unexpected = valid.clone();
        unexpected["vacate_request"]["extra"] = serde_json::json!(true);
        assert!(payload_identity(&unexpected).is_err());
    }

    #[test]
    fn motion_intent_v4_vacate_request_values_are_bounded() {
        let malformed = [
            serde_json::json!({
                "vehicle_id": "invalid vehicle",
                "cell": {"gx": 1, "gy": 0},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": 1, "gy": 0}]
            }),
            serde_json::json!({
                "vehicle_id": true,
                "cell": {"gx": 1, "gy": 0},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": 1, "gy": 0}]
            }),
            serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": true, "gy": 0},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": 1, "gy": 0}]
            }),
            serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1_000_001, "gy": 0},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": 1, "gy": 0}]
            }),
            serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": 1, "gy": 0}]
            }),
            serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1, "gy": 0, "extra": 0},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": 1, "gy": 0}]
            }),
            serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1, "gy": 0},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": true, "gy": 0}]
            }),
            serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1, "gy": 0},
                "route_cells": [{"gx": 0, "gy": 0}, {"gx": 1_000_001, "gy": 0}]
            }),
        ];
        for request in malformed {
            assert!(payload_identity(&motion_intent_v4(request)).is_err());
        }
    }

    #[test]
    fn motion_intent_v4_vacate_route_is_a_bounded_contiguous_array() {
        let request = |route_cells| {
            motion_intent_v4(serde_json::json!({
                "vehicle_id": "vehicle_2",
                "cell": {"gx": 1, "gy": 0},
                "route_cells": route_cells
            }))
        };
        assert!(
            payload_identity(&request(serde_json::json!([
                {"gx": 0, "gy": 0},
                {"gx": 0, "gy": 2}
            ])))
            .is_ok()
        );

        let too_long = (0..65)
            .map(|gx| serde_json::json!({"gx": gx, "gy": 0}))
            .collect::<Vec<_>>();
        for route_cells in [
            Value::Null,
            serde_json::json!({}),
            serde_json::json!([]),
            serde_json::json!([{"gx": 0, "gy": 0}]),
            Value::Array(too_long),
            serde_json::json!([{"gx": 0, "gy": 0}, {"gx": 0, "gy": 0}]),
            serde_json::json!([{"gx": 0, "gy": 0}, {"gx": 3, "gy": 0}]),
            serde_json::json!([{"gx": 1, "gy": 0}, {"gx": 2, "gy": 0}]),
        ] {
            assert!(payload_identity(&request(route_cells)).is_err());
        }
    }
}
